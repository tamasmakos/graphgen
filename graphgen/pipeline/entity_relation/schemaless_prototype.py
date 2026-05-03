from __future__ import annotations

from typing import Any, Dict, List, Tuple

from graphgen.pipeline.entity_relation.extractors import DSPyExtractor
from graphgen.utils.utils import standardize_label


RELATION_NORMALIZATION_MAP = {
    "HAS_SON": "PARENT_OF",
    "PARENTOF": "PARENT_OF",
    "COMPARED_TO": "IS_SIMILAR_TO",
    "PLAYED_WITH": "IS_PLAYED_ON",
    "HAS_NUMBER_OF": "HAS",
    "BEING_WATCHED_BY": "LOOKS_AT",
    "WORKS_AT": "AT",
}


def _triplet_to_relation(triplet: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    source = standardize_label(triplet.get("source", ""))
    relation = standardize_label(triplet.get("relation", ""))
    relation = RELATION_NORMALIZATION_MAP.get(relation, relation)
    target = standardize_label(triplet.get("target", ""))
    props = {
        "confidence": float(triplet.get("confidence", 1.0) or 1.0),
        "evidence": triplet.get("evidence", ""),
        "source_type": triplet.get("source_type", ""),
        "target_type": triplet.get("target_type", ""),
    }
    return source, relation, target, props


def postprocess_triplets(raw_triplets: List[Dict[str, Any]]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    relations: List[Tuple[str, str, str, Dict[str, Any]]] = []
    seen = set()

    for triplet in raw_triplets or []:
        source, relation, target, props = _triplet_to_relation(triplet)
        if not (source and relation and target):
            continue

        key = (source, relation, target)
        if key not in seen:
            relations.append((source, relation, target, props))
            seen.add(key)

        target_text = str(triplet.get("target", "")).strip()
        if relation == "IS" and " at " in target_text.lower():
            _, location_text = target_text.rsplit(" at ", 1)
            location_label = standardize_label(location_text)
            if location_label:
                derived_props = {
                    **props,
                    "target_type": standardize_label(props.get("derived_target_type") or "LOCATION"),
                }
                derived_key = (source, "AT", location_label)
                if derived_key not in seen:
                    relations.append((source, "AT", location_label, derived_props))
                    seen.add(derived_key)

    return relations


def build_nodes_from_relations(relations: List[Tuple[str, str, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    nodes = {}
    for source, _, target, props in relations:
        nodes.setdefault(source, {"id": source, "type": standardize_label(props.get("source_type") or "ENTITY"), "properties": {}})
        nodes.setdefault(target, {"id": target, "type": standardize_label(props.get("target_type") or "ENTITY"), "properties": {}})
    return list(nodes.values())


async def extract_schemaless_relations(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    extractor = DSPyExtractor(config)
    relations, nodes, diagnostics = await extractor.extract_relations(text=text, entities=[], abstract_concepts=[])

    if relations:
        return {
            "relations": relations,
            "nodes": nodes,
            "diagnostics": diagnostics,
        }

    raw_triplets = diagnostics.get("raw_triplets", []) if isinstance(diagnostics, dict) else []
    fallback_relations = postprocess_triplets(raw_triplets)
    fallback_nodes = build_nodes_from_relations(fallback_relations)
    return {
        "relations": fallback_relations,
        "nodes": fallback_nodes,
        "diagnostics": diagnostics,
    }
