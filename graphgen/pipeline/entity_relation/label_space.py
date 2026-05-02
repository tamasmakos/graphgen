from __future__ import annotations

from typing import Any, Dict, List

import logging
import numpy as np

from graphgen.utils.utils import standardize_label
from graphgen.utils.vector_embedder.model import get_model


logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return [token for token in standardize_label(text).lower().split('_') if token]


def _default_aliases(label: str) -> List[str]:
    pretty = standardize_label(label).lower().replace('_', ' ')
    aliases = {pretty}
    aliases.update(token for token in pretty.split() if len(token) > 2)
    return sorted(aliases)


def _default_description(label: str, parent_labels: List[str] | None = None) -> str:
    pretty = standardize_label(label).lower().replace('_', ' ')
    if parent_labels:
        parent_text = ', '.join(parent.lower().replace('_', ' ') for parent in parent_labels)
        return f"Named entity or concept labeled {pretty}; related to {parent_text}."
    return f"Named entity or concept labeled {pretty}."


def build_label_profiles(label_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for entry in label_entries or []:
        standardized = standardize_label(entry.get('label', ''))
        if not standardized:
            continue
        aliases = set(_default_aliases(standardized))
        aliases.update(str(alias).strip().lower() for alias in entry.get('aliases', []) if str(alias).strip())
        parent_labels = [standardize_label(parent) for parent in entry.get('parent_labels', []) if parent]
        description = (entry.get('description') or '').strip() or _default_description(standardized, parent_labels)
        profiles[standardized] = {
            'label': standardized,
            'aliases': sorted(aliases),
            'description': description,
            'source': entry.get('source', 'config'),
            'parent_labels': parent_labels,
        }
    return sorted(profiles.values(), key=lambda item: item['label'])


def _profile_text(profile: Dict[str, Any]) -> str:
    return ' '.join([
        profile['label'].lower().replace('_', ' '),
        profile.get('description', ''),
        ' '.join(parent.lower().replace('_', ' ') for parent in profile.get('parent_labels', [])),
        'aliases: ' + ', '.join(profile.get('aliases', [])),
    ]).strip()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_candidate_label_profiles(
    text: str,
    label_profiles: List[Dict[str, Any]],
    top_k: int = 5,
    strategy: str = 'hybrid',
) -> Dict[str, Any]:
    if not label_profiles:
        return {
            'candidate_labels': [],
            'candidate_profiles': [],
            'profile_scores': [],
        }

    lowered = text.lower()
    text_tokens = set(_tokenize(text))
    lexical_scores = []
    for profile in label_profiles:
        alias_tokens = set()
        lexical_score = 0.0
        for alias in profile.get('aliases', []):
            alias_lower = alias.lower()
            if alias_lower in lowered:
                lexical_score += max(1.0, len(alias_lower.split()))
            alias_tokens.update(token for token in alias_lower.split() if token)
        overlap = text_tokens & alias_tokens
        lexical_score += float(len(overlap))
        lexical_scores.append(lexical_score)

    semantic_scores = [0.0] * len(label_profiles)
    if strategy != 'lexical':
        try:
            model = get_model()
            if getattr(model, 'is_available', False):
                vectors = model.encode([text, *[_profile_text(profile) for profile in label_profiles]])
                query_vector = np.asarray(vectors[0], dtype=float)
                for idx, vector in enumerate(vectors[1:]):
                    semantic_scores[idx] = _cosine_similarity(query_vector, np.asarray(vector, dtype=float))
        except Exception as e:
            logger.warning("Semantic label selection failed; falling back to lexical scoring: %s", e)
            semantic_scores = [0.0] * len(label_profiles)

    scored = []
    for idx, profile in enumerate(label_profiles):
        lexical_score = lexical_scores[idx]
        semantic_score = semantic_scores[idx]
        if strategy == 'lexical':
            score = lexical_score
        elif strategy == 'embedding':
            score = semantic_score
        else:
            score = semantic_score + 0.35 * lexical_score
        scored.append({
            'label': profile['label'],
            'profile': profile,
            'score': score,
            'semantic_score': semantic_score,
            'lexical_score': lexical_score,
        })

    ranked = sorted(scored, key=lambda item: (-item['score'], item['label']))[: max(1, top_k)]
    return {
        'candidate_labels': [item['label'] for item in ranked],
        'candidate_profiles': [item['profile'] for item in ranked],
        'profile_scores': ranked,
    }


def build_gliner2_schema(candidate_profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    return {
        'entities': {
            profile['label'].lower(): profile.get('description') or _default_description(profile['label'], profile.get('parent_labels', []))
            for profile in candidate_profiles
        }
    }
