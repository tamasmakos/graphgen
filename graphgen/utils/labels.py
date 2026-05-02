"""
GLiNER Label Resolution.

Resolves GLiNER entity extraction labels from configuration,
supporting manual labels, ontology-based extraction, or both.
No default/fallback labels are injected.
"""

from pathlib import Path
import json
import logging
from typing import List, Dict, Any

from graphgen.utils.utils import standardize_label
from graphgen.utils.ontology_parser import extract_ontology_profiles

logger = logging.getLogger(__name__)


def _default_profile(label: str, source: str = "default") -> Dict[str, Any]:
    standardized = standardize_label(label)
    pretty = standardized.lower().replace("_", " ")
    aliases = sorted({pretty, *[token for token in pretty.split() if len(token) > 2]})
    return {
        "label": standardized,
        "aliases": aliases,
        "description": f"Named entity or concept labeled {pretty}.",
        "source": source,
    }


def _load_profiles_from_file(path: str) -> List[Dict[str, Any]]:
    profile_path = Path(path)
    if not profile_path.exists():
        logger.warning("Label profile file not found: %s", profile_path)
        return []

    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load label profiles from %s: %s", profile_path, exc)
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    logger.warning("Label profile file %s did not contain a list payload", profile_path)
    return []


def resolve_entity_label_profiles(extraction_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    manual_labels = extraction_config.get('entity_labels') or extraction_config.get('gliner_labels') or []
    description_overrides = {
        standardize_label(label): str(description).strip()
        for label, description in (extraction_config.get('gliner2_label_descriptions') or {}).items()
        if str(description).strip()
    }

    def _apply_override(profile: Dict[str, Any]) -> Dict[str, Any]:
        override = description_overrides.get(profile['label'])
        if not override:
            return profile
        return {**profile, 'description': override}

    ontology_config = extraction_config.get('ontology', {})
    if hasattr(ontology_config, 'model_dump'):
        ontology_config = ontology_config.model_dump()

    file_profiles = []
    label_profiles_path = extraction_config.get('label_profiles_path')
    if label_profiles_path:
        file_profiles = [_apply_override(profile) for profile in _load_profiles_from_file(label_profiles_path)]

    profiles = [_apply_override(_default_profile(label, source="config")) for label in manual_labels]
    if file_profiles:
        merged = {profile['label']: profile for profile in profiles}
        for profile in file_profiles:
            standardized = standardize_label(profile.get('label', ''))
            if not standardized:
                continue
            merged[standardized] = {
                'label': standardized,
                'aliases': sorted({*(profile.get('aliases') or []), standardized.lower().replace('_', ' ')}),
                'description': (profile.get('description') or '').strip() or _default_profile(standardized)['description'],
                'source': profile.get('source', 'file'),
                'parent_labels': [standardize_label(parent) for parent in profile.get('parent_labels', []) if parent],
            }
        profiles = sorted(merged.values(), key=lambda item: item['label'])
    ontology_enabled = ontology_config.get('enabled', False)
    if ontology_enabled:
        ontology_profiles = extract_ontology_profiles(
            ontology_dir=ontology_config.get('ontology_dir', 'input/ontology/cdm-4.13.2'),
            namespace_filter=ontology_config.get('namespace_filter'),
            include_local_names=ontology_config.get('include_local_names', True),
            top_level_only=ontology_config.get('top_level_only', True),
            min_subclasses=ontology_config.get('min_subclasses', 0),
        )
        if ontology_profiles:
            ontology_profiles = [_apply_override(profile) for profile in ontology_profiles]
            if ontology_config.get('merge_with_manual', True):
                merged = {profile['label']: profile for profile in profiles}
                for profile in ontology_profiles:
                    merged[profile['label']] = profile
                profiles = sorted(merged.values(), key=lambda item: item['label'])
            else:
                profiles = ontology_profiles

    return profiles


def resolve_entity_labels(extraction_config: Dict[str, Any]) -> List[str]:
    """
    Resolve GLiNER labels from configuration.
    
    Supports multiple sources for entity labels:
    1. Manual labels from config (gliner_labels)
    2. Ontology-derived labels from RDF/OWL files
    3. Merged combination of both sources
    
    Args:
        extraction_config: Extraction configuration dictionary.
                          Expected to have 'gliner_labels' and optionally 'ontology' keys.
    
    Returns:
        List of unique entity labels for GLiNER extraction.
    """
    profiles = resolve_entity_label_profiles(extraction_config)
    return [profile['label'] for profile in profiles]


def _extract_ontology_labels(ontology_config: Dict[str, Any]) -> List[str]:
    """
    Extract labels from ontology based on configuration.
    
    Args:
        ontology_config: Ontology configuration dictionary.
        
    Returns:
        List of labels extracted from ontology, or empty list on failure.
    """
    try:
        from graphgen.utils.ontology_parser import OntologyLabelExtractor
        
        ontology_dir = ontology_config.get('ontology_dir', 'input/ontology/cdm-4.13.2')
        namespace_filter = ontology_config.get('namespace_filter')
        include_local_names = ontology_config.get('include_local_names', True)
        top_level_only = ontology_config.get('top_level_only', True)
        min_subclasses = ontology_config.get('min_subclasses', 0)
        
        logger.info(f"Extracting labels from ontology: {ontology_dir} (top_level={top_level_only}, min_children={min_subclasses})")
        
        extractor = OntologyLabelExtractor(
            ontology_dir=ontology_dir,
            namespace_filter=namespace_filter
        )
        
        labels = extractor.extract_labels(
            include_local_names=include_local_names,
            top_level_only=top_level_only,
            min_subclasses=min_subclasses
        )
        
        logger.info(f"Extracted {len(labels)} labels from ontology")
        return labels
        
    except ImportError as e:
        logger.error(f"Failed to import ontology parser: {e}")
        return []
    except Exception as e:
        logger.error(f"Ontology label extraction failed: {e}")
        return []
