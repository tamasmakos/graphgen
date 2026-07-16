"""
Unified Entity Resolution Module.

This module handles:
1. Fast string-based coreference resolution (for initial extraction).
2. Multi-strategy global entity resolution (for graph refinement):
   - Stage 1: Deterministic Acronym/Abbreviation Resolution
   - Stage 2: Substring Containment Resolution
   - Stage 3: Embedding-Based Semantic Resolution
   - Stage 4: Title/Role Coreference Resolution
"""

import logging
import re
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from difflib import SequenceMatcher
from collections import defaultdict
from itertools import combinations

from graphgen.utils.utils import merge_node_into, get_spacy_model

logger = logging.getLogger(__name__)

# --- Part 1: String-Based Helpers & Lightweight Extraction Coref ---

def _canonicalize_entity_name(name: str) -> str:
    """
    Canonicalize entity name:
    - Lowercase
    - Remove punctuation
    - Singularize (simple heuristic)
    """
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    return name

def _get_acronym_candidates(text: str) -> List[str]:
    """
    Generate potential acronyms for a text.
    E.g. "European_Union" -> ["EU"]
    """
    if not text:
        return []
    
    # Split by non-word chars OR underscores
    tokens = [t for t in re.split(r'[\W_]+', text) if t]
    if len(tokens) < 2:
        return []
        
    # Standard First-Letter Acronym
    acronym = "".join([t[0].upper() for t in tokens])
    return [acronym]

def _is_acronym_match(short: str, long: str) -> bool:
    """
    Check if 'short' is a valid acronym for 'long'.
    """
    # 1. Short must be all clean uppercase and short-ish
    clean_short = short.strip().replace('.', '').replace(' ', '')
    if not clean_short.isupper() or len(clean_short) > 6 or len(clean_short) < 2:
        return False
        
    # 2. Generate candidates from long
    candidates = _get_acronym_candidates(long)
    return clean_short in candidates

def _is_numeric_entity(name: str) -> bool:
    """
    Return True if an entity name is primarily a numeric/quantitative value.
    E.g. "150", "2_5_MILLION", "0_2_PERCENT", "30_BILLION".
    These should never be semantically merged regardless of embedding similarity.
    """
    numeric_suffixes = {'BILLION', 'MILLION', 'THOUSAND', 'PERCENT', 'TRILLION', 'HUNDRED'}
    tokens = name.split('_')
    if not tokens:
        return False
    has_digit_token = any(re.match(r'^\d+$', t) or re.match(r'^\d+[\.,]\d+$', t) for t in tokens)
    all_numeric_tokens = all(
        re.match(r'^\d+$', t) or
        re.match(r'^\d+[\.,]\d+$', t) or
        t in numeric_suffixes
        for t in tokens
    )
    return has_digit_token and all_numeric_tokens


def _is_safe_substring_merge(
    child: str,
    parent: str,
    child_degree: int = 0,
    parent_degree: int = 0
) -> bool:
    """
    Check if merging 'child' into 'parent' is semantically safe.

    Three layered guards, ordered cheapest-to-most-expensive:

    Guard 1 — Long-phrase absorption:
        A single-token child being absorbed by a phrase with 3+ extra tokens is
        almost always a generic common noun swallowed by a specific description
        (e.g. WORK -> WORK_OF_THE_CONFERENCE_ON_THE_FUTURE_OF_EUROPE).
        Only allow it if the child is a short uppercase acronym (≤5 chars) with
        low independent usage.

    Guard 2 — Adjective-qualifier prefix/suffix:
        If the parent is "QUALIFIER_CHILD" or "CHILD_QUALIFIER" where the extra
        token is an adjective or determiner, the parent is a *description* of the
        child, not a canonical form of it (e.g. STRONG_NATO, POLITICAL_COURAGE,
        WINNING_STRATEGY). Only allow this merge when the child is barely used
        standalone (degree ≤ 2) AND the parent is significantly more central.

    Guard 3 — Degree-based importance:
        If the child node is equally or more connected than the parent, it is the
        primary concept in the graph. Absorbing it into the parent would destroy
        information. Require parent to be strictly more central.
    """
    try:
        child_tokens = child.split('_')
        parent_tokens = parent.split('_')
        child_token_set = set(child_tokens)
        extra_tokens = [t for t in parent_tokens if t not in child_token_set]

        # Guard 1: long-phrase absorption of a single common word
        if len(child_tokens) == 1 and len(extra_tokens) >= 3:
            # Allow acronym-like strings (short, already all-caps, rarely used alone)
            is_acronym_like = len(child) <= 5
            if not is_acronym_like or child_degree > 2:
                return False

        # Guard 2: POS-based qualifier check (requires spaCy)
        nlp = get_spacy_model()
        if nlp and len(child_tokens) == 1 and len(extra_tokens) == 1:
            extra_word = extra_tokens[0].replace('_', ' ')
            # Normalize to title-case for better spaCy tagging
            doc = nlp(extra_word.capitalize())
            if doc:
                extra_pos = doc[0].pos_
                if extra_pos in ('ADJ', 'DET', 'NUM', 'VERB'):
                    # Parent is "MODIFIER + CHILD" or "CHILD + MODIFIER"
                    # Only safe if child is obscure (barely connected) and
                    # parent is clearly more central
                    child_is_obscure = child_degree <= 2
                    parent_is_dominant = parent_degree > child_degree * 2
                    if not (child_is_obscure and parent_is_dominant):
                        return False

        # Guard 3: degree-based importance
        # If child is at least as connected as parent, child is the primary concept.
        if child_degree > 0 and parent_degree > 0:
            if child_degree >= parent_degree:
                return False

        # Original root-head check: child should cover the syntactic head of parent
        if not nlp:
            return True

        raw_text = parent.replace('_', ' ').strip()
        if not raw_text:
            return True

        parent_text = raw_text.title() if raw_text.isupper() else raw_text
        doc = nlp(parent_text)

        root = next((token for token in doc if token.head == token), None)
        if not root:
            return True

        root_text = root.text.lower()
        root_lemma = root.lemma_.lower()
        child_clean = child.replace('_', ' ').lower()

        if root_text not in child_clean and root_lemma not in child_clean:
            return False

        return True
    except Exception:
        return True

def resolve_extraction_coreferences(
    relations: List[Tuple[str, str, str, Dict[str, Any]]], 
    entities: List[str],
    similarity_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Lightweight entity normalization for raw extraction data.
    """
    try:
        # 1) Collect all surface forms
        originals: Set[str] = set()
        for i, item in enumerate(relations or []):
            if len(item) >= 3:
                s, _, t = item[0], item[1], item[2]
                if isinstance(s, str): originals.add(s)
                if isinstance(t, str): originals.add(t)
        for e in entities or []:
            if isinstance(e, str): originals.add(e)

        # 2) Initial canonicalization for grouping
        orig_to_canon: Dict[str, str] = {o: _canonicalize_entity_name(o) for o in originals}
        canonicals: List[str] = sorted(set(orig_to_canon.values()))

        # 3) Greedy grouping
        rep_for: Dict[str, str] = {}
        representatives: List[str] = []
        
        for c in canonicals:
            placed = False
            for r in representatives:
                # Similarity match
                is_match = False
                
                # A. String Similarity
                if SequenceMatcher(None, c, r).ratio() >= similarity_threshold:
                    is_match = True
                
                # B. Acronym Match (Simple)
                # We need original cases for acronyms, but we only have canonicals here (lowercased)
                # So we skip strict acronym check here and rely on Stage 1 in global resolution
                
                if is_match:
                    # Choose longer string as representative
                    best = r if len(r) >= len(c) else c
                    
                    if best != r:
                        # Update references to old Rep
                        for k, v in list(rep_for.items()):
                            if v == r:
                                rep_for[k] = best
                        representatives[representatives.index(r)] = best
                        
                    rep_for[c] = representatives[representatives.index(best)]
                    placed = True
                    break
            
            if not placed:
                representatives.append(c)
                rep_for[c] = c

        # 4) Final mapping
        canon_to_best_original = {}
        for r in representatives:
            candidates = [o for o, c in orig_to_canon.items() if rep_for.get(c) == r]
            if candidates:
                best_orig = sorted(candidates, key=lambda x: (len(x), sum(1 for c in x if c.isupper())), reverse=True)[0]
                canon_to_best_original[r] = best_orig

        entity_mappings: Dict[str, str] = {}
        for o, c in orig_to_canon.items():
            rep_canon = rep_for.get(c, c)
            final_name = canon_to_best_original.get(rep_canon, o)
            entity_mappings[o] = final_name

        # 5) Remap relations
        cleaned_list: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for item in relations or []:
            if len(item) == 4:
                s, r, t, props = item
            else:
                s, r, t = item[0], item[1], item[2]
                props = {}
                
            cs = entity_mappings.get(s, s)
            ct = entity_mappings.get(t, t)
            if not cs or not ct or cs == ct:
                continue
            cleaned_list.append((cs, r, ct, props))

        return {
            'cleaned_relations': cleaned_list,
            'entity_mappings': entity_mappings,
        }
    except Exception as e:
        logger.warning(f"Lightweight coreference normalization failed: {e}")
        return {'cleaned_relations': relations, 'entity_mappings': {}}


# --- Part 2: Global Graph Resolution (Multi-Strategy) ---

def resolve_entities_semantically(
    graph: nx.DiGraph,
    similarity_threshold: float = 0.85, # Default lowered slightly for embeddings
    node_types: Optional[List[str]] = None,
    structural_embeddings: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Multi-stage entity resolution pipeline.
    """
    if node_types is None:
        node_types = ['ENTITY_CONCEPT']
        
    logger.info("Starting Multi-Strategy Entity Resolution...")
    
    # Collect candidates
    nodes = []
    for n, d in graph.nodes(data=True):
        if d.get('node_type') in node_types:
            nodes.append(n)
            
    if len(nodes) < 2:
        return {'merged_nodes': 0}
        
    stats = {
        'initial_count': len(nodes),
        'merges_stage_1_acronym': 0,
        'merges_stage_2_containment': 0,
        'merges_stage_3_semantic': 0,
        'removed_stage_4_titles': 0,
        'merged_pairs_details': []
    }
    
    # --- STAGE 1: Acronym/Abbreviation Resolution ---
    # O(N^2) worst case but N is usually small (<1000 entities per batch)
    # We can optimize by separating short vs long entities
    short_entities = [n for n in nodes if len(n) <= 6 and n.isupper()]
    long_entities = [n for n in nodes if len(n) > 6]
    
    # Sort short entities by length desc to handle dependencies
    short_entities.sort(key=len, reverse=True)
    
    for short in short_entities:
        if not graph.has_node(short): continue
        
        best_match = None
        
        # Check against all long entities
        candidates = []
        for long in long_entities:
            if not graph.has_node(long): continue
            
            if _is_acronym_match(short, long):
                candidates.append(long)
                
        if candidates:
            # Pick best: most connected or default to first
            candidates.sort(key=lambda x: graph.degree(x), reverse=True)
            best_match = candidates[0]

            # Degree guard: if the short form is more connected than the long form,
            # the short form is the real primary entity in this corpus (e.g. "IT" as
            # a well-used node is more likely a country code than a coincidental
            # acronym of some rarely-mentioned phrase).
            if graph.degree(short) > graph.degree(best_match):
                logger.info(
                    f"[Stage 1] Skipping acronym merge {short} -> {best_match}: "
                    f"short is more central (degree {graph.degree(short)} vs {graph.degree(best_match)})"
                )
                continue

            # Merge Short -> Long
            logger.info(f"[Stage 1] Merging Acronym {short} -> {best_match}")
            merge_node_into(graph, short, best_match)
            stats['merges_stage_1_acronym'] += 1
            stats['merged_pairs_details'].append(f"ACRONYM: {short} -> {best_match}")
            
    # Refresh node list after merges
    nodes = [n for n in nodes if graph.has_node(n)]


    # --- STAGE 2: Substring Containment ---
    # "PARLIAMENT" -> "EUROPEAN_PARLIAMENT"
    # ONLY if Unambiguous (contained in exactly one parent)
    
    # Sort by length ASC (shortest first)
    nodes.sort(key=len)
    
    for i, child in enumerate(nodes):
        if not graph.has_node(child): continue
        # Skip if child is very generic or long
        if len(child.split('_')) > 3: continue
        
        child_tokens = set(child.split('_'))
        if len(child_tokens) == 0: continue
        
        parents = []
        for j in range(i + 1, len(nodes)):
            parent = nodes[j]
            if not graph.has_node(parent): continue
            
            parent_tokens = set(parent.split('_'))
            if child_tokens.issubset(parent_tokens) and len(parent_tokens) > len(child_tokens):
                parents.append(parent)
                
        if len(parents) == 1:
            # Unambiguous containment
            parent = parents[0]
            child_deg = graph.degree(child)
            parent_deg = graph.degree(parent)

            if not _is_safe_substring_merge(child, parent, child_deg, parent_deg):
                logger.info(f"[Stage 2] Skipping unsafe merge {child} -> {parent} (safety check failed)")
                continue
                
            logger.info(f"[Stage 2] Merging Substring {child} -> {parent}")
            merge_node_into(graph, child, parent)
            stats['merges_stage_2_containment'] += 1
            stats['merged_pairs_details'].append(f"SUBSTRING: {child} -> {parent}")
            
        elif len(parents) > 1:
            # Ambiguous: "UNION" -> "EUROPEAN_UNION", "TRANSFER_UNION"
            # Check connectivity
            parents.sort(key=lambda x: graph.degree(x), reverse=True)
            winner = parents[0]
            runner_up = parents[1]

            deg_w = graph.degree(winner)
            deg_r = graph.degree(runner_up)
            child_deg = graph.degree(child)

            # Only merge if winner is dominant (e.g. degree 10x higher)
            if deg_w > 5 and deg_w > (deg_r * 3) and _is_safe_substring_merge(child, winner, child_deg, deg_w):
                logger.info(f"[Stage 2] Merging Ambiguous Substring {child} -> {winner} (Dominant Parent)")
                merge_node_into(graph, child, winner)
                stats['merges_stage_2_containment'] += 1
                stats['merged_pairs_details'].append(f"SUBSTRING_DOMINANT: {child} -> {winner}")

    # Refresh node list
    nodes = [n for n in nodes if graph.has_node(n)]

    # --- STAGE 3: Semantic Embedding Resolution ---
    # Collect embeddings
    valid_nodes = []
    embeddings = []
    struc_embeddings = []
    
    for n in nodes:
        emb = graph.nodes[n].get('embedding')
        if emb is not None and isinstance(emb, np.ndarray):
            valid_nodes.append(n)
            embeddings.append(emb)
            
            if structural_embeddings and n in structural_embeddings:
                struc_embeddings.append(structural_embeddings[n])
            else:
                struc_embeddings.append(None)
                
    if len(valid_nodes) > 1:
        embeddings = np.array(embeddings)
        
        # Helper for combined similarity
        def calc_similarity(idx_a, idx_b):
            # Semantic
            vec_a, vec_b = embeddings[idx_a], embeddings[idx_b]
            sim_sem = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            
            # Subcomponent 2: String (Jaccard on tokens)
            name_a, name_b = valid_nodes[idx_a], valid_nodes[idx_b]
            tokens_a = set(name_a.split('_'))
            tokens_b = set(name_b.split('_'))
            inter = len(tokens_a.intersection(tokens_b))
            union = len(tokens_a.union(tokens_b))
            sim_str = inter / union if union > 0 else 0.0
            
            # Subcomponent 3: Structural
            sim_struc = 0.0
            has_structural = False
            if struc_embeddings[idx_a] is not None and struc_embeddings[idx_b] is not None:
                s_a, s_b = struc_embeddings[idx_a], struc_embeddings[idx_b]
                sim_struc = np.dot(s_a, s_b) / (np.linalg.norm(s_a) * np.linalg.norm(s_b))
                has_structural = True
            
            # Weighted Score
            if has_structural:
                # 60% Semantic, 20% Structural, 20% String
                score = (0.6 * sim_sem) + (0.2 * sim_struc) + (0.2 * sim_str)
            else:
                # Redistribute structural weight to semantic
                # 75% Semantic + 25% String
                score = (0.75 * sim_sem) + (0.25 * sim_str)
                
            return score, sim_sem
            
        # Pairwise (Blocking optimization skipped for prototype size, just O(N^2) for <500 nodes is fine)
        merged_in_this_pass = set()
        
        for i in range(len(valid_nodes)):
            if valid_nodes[i] in merged_in_this_pass: continue
            
            best_match_idx = -1
            best_score = -1.0
            
            for j in range(i + 1, len(valid_nodes)):
                if valid_nodes[j] in merged_in_this_pass: continue

                # Numeric guard: quantities like "150_BILLION" and "30_BILLION" have
                # structurally similar embeddings but represent entirely different values.
                # Never merge two numeric/quantitative entities regardless of similarity.
                if _is_numeric_entity(valid_nodes[i]) or _is_numeric_entity(valid_nodes[j]):
                    continue
                
                score, raw_sem = calc_similarity(i, j)
                
                # Thresholds
                # 0.82 Combined Score OR 0.90 Raw Semantic
                if score > 0.82 or raw_sem > 0.92:
                    if score > best_score:
                        best_score = score
                        best_match_idx = j
            
            if best_match_idx != -1:
                node_a = valid_nodes[i]
                node_b = valid_nodes[best_match_idx]
                
                # Determine direction: merge less connected into more connected
                if graph.degree(node_b) > graph.degree(node_a):
                    source, target = node_a, node_b
                else:
                    source, target = node_b, node_a
                    
                logger.info(f"[Stage 3] Merging Semantic {source} -> {target} (Score={best_score:.3f})")
                merge_node_into(graph, source, target)
                merged_in_this_pass.add(source) # Mark as gone
                stats['merges_stage_3_semantic'] += 1
                stats['merged_pairs_details'].append(f"SEMANTIC: {source} -> {target}")

    # --- STAGE 4: Title/Role Cleanup ---
    title_indicators = ["MR_", "MRS_", "MS_", "MADAM_", "PRESIDENT", "SPEAKER", "MINISTER", "COMMISSIONER"]
    
    # Refresh nodes
    for n in list(graph.nodes()):
        if graph.nodes[n].get('node_type') == 'ENTITY_CONCEPT':
            is_title = False
            for ind in title_indicators:
                if n.startswith(ind) or n.endswith(f"_{ind}"):
                    is_title = True
                    break
            
            if is_title:
                # Ambiguous role/title
                degree = graph.degree(n)
                if degree <= 2:
                    # Low connectivity title -> unlikely to be resolved -> Remove
                    logger.info(f"[Stage 4] Removing ambiguous title node: {n}")
                    graph.remove_node(n)
                    stats['removed_stage_4_titles'] += 1
                else:
                    # High connectivity -> Flag as role
                    graph.nodes[n]['is_role'] = True

    final_count = len([n for n in graph.nodes() if graph.nodes[n].get('node_type') in node_types])
    stats['final_count'] = final_count
    
    logger.info(f"Resolution Complete. {stats['merges_stage_1_acronym']} acronyms, {stats['merges_stage_2_containment']} substrings, {stats['merges_stage_3_semantic']} semantic merges.")
    
    return stats
