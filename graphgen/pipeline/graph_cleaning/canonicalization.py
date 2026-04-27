"""Conservative entity surface-form canonicalization helpers.

These helpers are intentionally modest. They aim to improve graph cleanliness
without introducing broad ontology matching or aggressive semantic collapse.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

ROLE_TERMS = {
    "PRIME_MINISTER",
    "PRESIDENT",
    "CHANCELLOR",
    "COMMISSIONER",
    "MINISTER",
    "SPEAKER",
    "CHAIR",
    "LEADER",
}

CONCEPT_LIKE_TERMS = {
    "MIGRATION",
    "HEALTH",
    "CLIMATE",
    "SECURITY",
    "SANCTIONS",
    "AID",
    "POLICY_CHANGES",
    "WHATEVER_IT_TAKES",
}

NAMED_ENTITY_HINTS = {
    "EUROPEAN_CENTRAL_BANK",
    "EUROPEAN_PARLIAMENT",
    "MARIO_DRAGHI",
    "ROBERTA_METSOLA",
    "UKRAINE",
    "ITALY",
    "EU",
    "EUROPE",
}


def normalize_surface_form(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).strip().upper()
    normalized = re.sub(r"[^A-Z0-9\s_]", " ", normalized)
    normalized = re.sub(r"[\s_]+", "_", normalized).strip("_")
    return normalized


def classify_surface_form(text: str) -> str:
    normalized = normalize_surface_form(text)
    if not normalized:
        return "unknown"
    if normalized in ROLE_TERMS:
        return "role_artifact"
    if normalized in CONCEPT_LIKE_TERMS:
        return "concept_like"
    if normalized in NAMED_ENTITY_HINTS:
        return "named_entity"
    if "_" in normalized and all(token[:1].isalpha() for token in normalized.split("_") if token):
        return "named_entity"
    if len(normalized) <= 3:
        return "unknown"
    return "concept_like"


def are_potential_aliases(a: str, b: str) -> bool:
    norm_a = normalize_surface_form(a)
    norm_b = normalize_surface_form(b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True

    class_a = classify_surface_form(norm_a)
    class_b = classify_surface_form(norm_b)
    if "role_artifact" in {class_a, class_b} and class_a != class_b:
        return False
    if {norm_a, norm_b} == {"EU", "EUROPE"}:
        return False

    ratio = SequenceMatcher(None, norm_a.replace("_", " "), norm_b.replace("_", " ")).ratio()
    if ratio >= 0.92:
        return True

    tokens_a = set(norm_a.split("_"))
    tokens_b = set(norm_b.split("_"))
    if tokens_a and tokens_b and tokens_a == tokens_b:
        return True

    return False
