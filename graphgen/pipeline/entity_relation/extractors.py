"""
Graph Extractor Abstraction Layer.

Provides a unified interface for different graph extraction backends:
- LangChain LLMGraphTransformer
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import logging
import asyncio
import time
# from gliner import GLiNER

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer

from graphgen.config.llm import get_langchain_llm, configure_dspy_lm
from graphgen.pipeline.entity_relation.dspy_module import (
    GraphExtractorModule,
    OntologyGuidedExtractorModule,
)
import dspy
import os

from graphgen.utils.utils import standardize_label
from graphgen.pipeline.entity_relation.dspy_module import ENTITY_LIST_SEPARATOR

logger = logging.getLogger(__name__)



DEFAULT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at extracting knowledge graph entities and relationships from text.    
    Text:
    {input}
    """
)

class BaseExtractor(ABC):
    """Base class for graph extractors."""

    # Whether this extractor consumes NER (GLiNER/Spacy) entity hints.
    # Extractors that do their own typing (e.g. ontology-guided) set this
    # False so the pipeline can skip the expensive, GPU-serialised NER step.
    requires_ner_hints: bool = True

    @abstractmethod
    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Extract relations from text.
        
        Args:
            text: Text to extract relations from
            custom_prompt: Optional custom prompt template
            keywords: Optional list of keywords to guide extraction
            entities: Optional list of entities to focus on (used by LangChain)
            abstract_concepts: Optional list of abstract concepts (used by LangChain)
            
            text: Text to extract relations from
            custom_prompt: Optional custom prompt template
            keywords: Optional list of keywords to guide extraction
            entities: Optional list of entities to focus on (used by LangChain)
            abstract_concepts: Optional list of abstract concepts (used by LangChain)
            
        Returns:
            Tuple containing:
            - List of (source, relation_type, target, properties) tuples
            - List of extracted nodes with metadata (id, type, properties)
        """
        pass
    
    async def close(self):
        """Cleanup resources."""
        pass


class LangChainExtractor(BaseExtractor):
    """LangChain LLMGraphTransformer-based extractor."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config
        # Initialize GLiNER for entity extraction
        # gliner_model = config.get('gliner_model', 'urchade/gliner_medium-v2.1')
        # self.gliner = GLiNER.from_pretrained(gliner_model)
        # self.entity_labels = config.get('entity_labels', ["person", "organization", "location", "event", "concept", "product", "date", "time"])
        # logger.info(f"Initialized LangChain extractor with GLiNER ({gliner_model})")
    
    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        """Extract relations using LangChain LLMGraphTransformer."""
        # abstract_concepts now contains the ontology labels (Types)
        allowed_nodes = abstract_concepts or []
        # entities contains the hints from GLiNER/Spacy
        discovered_hints = entities or []
        
        # Inject hints into the prompt if available
        hints_text = ""
        if discovered_hints:
            hints_text = f"\nPre-identified entities found in text: {', '.join(discovered_hints)}\n"
        
        prompt = custom_prompt or DEFAULT_EXTRACTION_PROMPT
        
        # We can wrap the prompt to include hints if needed, but for now let's use it as is
        # or append hints to the input text
        full_text = text
        if hints_text:
            full_text = f"{hints_text}\nInput Text:\n{text}"
        
        def _extract_sync():
            llm = get_langchain_llm(self.config, purpose='extraction')
            
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=allowed_nodes,
                prompt=prompt,
                strict_mode=True, # Set to True to strictly follow ontology
                node_properties=False,
                relationship_properties=False
            )
            
            document = Document(page_content=full_text)
            return transformer.convert_to_graph_documents([document])

        retries = 3
        retry_delay = 1
        
        for attempt in range(retries):
            try:
                # Run in executor to avoid blocking
                graph_docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _extract_sync
                )
                
                if not graph_docs:
                    return [], []
                
                # Extract triplets and nodes
                relations = []
                nodes_data = []
                
                for graph_doc in graph_docs:
                    # Extract relations
                    for relationship in graph_doc.relationships:
                        source = standardize_label(relationship.source.id)
                        target = standardize_label(relationship.target.id)
                        relation_type = standardize_label(relationship.type)
                        props = dict(relationship.properties or {})
                        relations.append((source, relation_type, target, props))
                        
                    # Extract nodes
                    for node in graph_doc.nodes:
                        nodes_data.append({
                            "id": standardize_label(node.id),
                            "type": standardize_label(node.type),
                            "properties": dict(node.properties or {})
                        })
                
                return relations, nodes_data
                
            except Exception as e:
                # Check for 400 Bad Request / Tool use failed
                error_str = str(e)
                if "400" in error_str or "tool_use_failed" in error_str or "BadRequest" in error_str:
                    logger.warning(f"LangChain extraction failed with 400/Tool Error (attempt {attempt+1}/{retries}): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        # Last retry for 400 error, return empty
                        logger.error(f"All {retries} retries exhausted for 400/Tool Error")
                        return [], []
                
                # For other errors, log and retry or return empty
                logger.error(f"LangChain extraction failed: {e}", exc_info=True)
                if attempt < retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    return [], []
        
        return [], []


def _parse_dspy_triplets(
    raw_triplets: list,
) -> tuple:
    """
    Convert a list of DSPy Triplet objects (or dicts) into the canonical
    ``(relations, nodes_data)`` format consumed by the rest of the pipeline.

    Fan-out handling
    ----------------
    When an LLM returns a JSON list for an entity field (e.g. target:
    ["RUSSIA", "CHINA", "TURKEY"]), the Triplet field_validator joins the
    values with ENTITY_LIST_SEPARATOR rather than crashing.  This function
    detects those delimited strings and performs a cartesian-product expansion
    so that one "multi-valued" triplet becomes N individual triplets.

    Examples
    --------
    Single target (normal case):
        Triplet(source="EU", relation="sanctions", target="RUSSIA", ...)
        → [("EU", "SANCTIONS", "RUSSIA", {...})]

    Multi-target (fan-out, after validator coercion):
        Triplet(source="EU", relation="sanctions", target="RUSSIA|||CHINA|||TURKEY", ...)
        → [("EU", "SANCTIONS", "RUSSIA", {...}),
           ("EU", "SANCTIONS", "CHINA",  {...}),
           ("EU", "SANCTIONS", "TURKEY", {...})]

    Multi-source AND multi-target (full cartesian product):
        Triplet(source="A|||B", relation="rel", target="X|||Y", ...)
        → [("A","REL","X",{...}), ("A","REL","Y",{...}),
           ("B","REL","X",{...}), ("B","REL","Y",{...})]

    Returns
    -------
    relations : list of (source, relation, target, props) tuples
    nodes_data : list of {"id": ..., "type": ..., "properties": {}} dicts
    """
    relations = []
    nodes_data = []
    seen_nodes: set = set()

    for triplet in (raw_triplets or []):
        if isinstance(triplet, dict):
            source = triplet.get("source") or ""
            relation = triplet.get("relation") or ""
            target = triplet.get("target") or ""
            source_type = triplet.get("source_type") or "ENTITY"
            target_type = triplet.get("target_type") or "ENTITY"
            confidence = triplet.get("confidence", 1.0)
            evidence = triplet.get("evidence", "")
        else:
            source = getattr(triplet, "source", "") or ""
            relation = getattr(triplet, "relation", "") or ""
            target = getattr(triplet, "target", "") or ""
            source_type = getattr(triplet, "source_type", None) or "ENTITY"
            target_type = getattr(triplet, "target_type", None) or "ENTITY"
            confidence = getattr(triplet, "confidence", 1.0)
            evidence = getattr(triplet, "evidence", "")

        if not source or not relation or not target:
            continue

        # Expand fan-out values back into individual entity strings
        sources = [s.strip() for s in source.split(ENTITY_LIST_SEPARATOR) if s.strip()]
        targets = [t.strip() for t in target.split(ENTITY_LIST_SEPARATOR) if t.strip()]

        relation = standardize_label(relation)
        source_type = standardize_label(source_type)
        target_type = standardize_label(target_type)

        props = {"confidence": confidence, "evidence": evidence}

        for src in sources:
            src = standardize_label(src)
            for tgt in targets:
                tgt = standardize_label(tgt)
                if not src or not tgt or src == tgt:
                    continue
                relations.append((src, relation, tgt, props))

                if src not in seen_nodes:
                    nodes_data.append({"id": src, "type": source_type, "properties": {}})
                    seen_nodes.add(src)
                if tgt not in seen_nodes:
                    nodes_data.append({"id": tgt, "type": target_type, "properties": {}})
                    seen_nodes.add(tgt)

    return relations, nodes_data


class DSPyExtractor(BaseExtractor):
    """DSPy-based extractor."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config
        model = configure_dspy_lm(config, purpose="extraction")
        self.module = GraphExtractorModule()
        logger.info(f"Initialized DSPy extractor with model {model}")

    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        """Extract relations using DSPy."""
        
        ontology_classes = abstract_concepts or []
        entity_hints = entities or []
        
        try:
            # DSPy calls are synchronous, so we run in a thread
            def _extract_sync():
                # The dspy module returns a Prediction object which has the output fields as attributes
                prediction = self.module(text=text, ontology_classes=ontology_classes, entity_hints=entity_hints)
                return prediction.triplets
                
            triplets = await asyncio.to_thread(_extract_sync)
            return _parse_dspy_triplets(triplets)
            
        except Exception as e:
            logger.error(f"DSPy extraction failed: {e}", exc_info=True)
            return [], []

class OntologyGuidedDSPyExtractor(BaseExtractor):
    """DSPy extractor that first matches each chunk to a single ontology.

    Per chunk, the text is embedded and compared against the in-memory
    ontology registry.  The best-matching ontology's description and label
    set are fed to a ``dspy.Predict`` signature so extraction is constrained
    to exactly one ontology's types.  Every emitted triplet is tagged with
    the matched ontology name for provenance.

    Ontology matching + the LLM assign entity types directly, so the GLiNER
    NER-hint step is redundant and is skipped for this backend.
    """

    requires_ner_hints = False

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model = configure_dspy_lm(config, purpose="extraction")
        self.module = OntologyGuidedExtractorModule()

        extraction_cfg = config.get('extraction', {})
        if hasattr(extraction_cfg, 'model_dump'):
            extraction_cfg = extraction_cfg.model_dump()
        ontology_cfg = extraction_cfg.get('ontology', {}) or {}

        embedding_cfg = config.get('embedding', {})
        if hasattr(embedding_cfg, 'model_dump'):
            embedding_cfg = embedding_cfg.model_dump()
        embed_model = embedding_cfg.get('model_name', 'all-MiniLM-L6-v2')

        from graphgen.pipeline.entity_relation.ontology_matcher import OntologyRegistry

        self.registry = OntologyRegistry(
            ontology_dir=ontology_cfg.get('ontology_dir', 'input/ontology/cdm-4.13.2'),
            embed_model_name=embed_model,
            top_level_only=ontology_cfg.get('top_level_only', True),
            min_subclasses=ontology_cfg.get('min_subclasses', 0),
            include_local_names=ontology_cfg.get('include_local_names', True),
            max_labels=ontology_cfg.get('max_labels', 60),
            match_threshold=ontology_cfg.get('match_threshold', 0.15),
        ).build()

        # Fallback labels when no ontology clears the match threshold.
        from graphgen.utils.labels import resolve_entity_labels
        self.fallback_labels = resolve_entity_labels(extraction_cfg)

        logger.info(
            "Initialized OntologyGuidedDSPyExtractor with model %s (%d ontologies).",
            model,
            len(self.registry.entries),
        )

    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None,
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        if not text or not text.strip():
            return [], []

        entry, sim = self.registry.match(text)
        if entry is not None:
            description = entry.description or entry.name
            labels = entry.labels
            ontology_name = entry.name
        else:
            # No ontology cleared the threshold — fall back to the merged label
            # set so extraction still runs, tagged as 'unmatched'.
            description = "General knowledge extraction (no specific ontology matched)."
            labels = abstract_concepts or self.fallback_labels
            ontology_name = None

        try:
            def _extract_sync():
                prediction = self.module(
                    text=text,
                    ontology_description=description,
                    ontology_labels=labels,
                )
                return prediction.triplets

            triplets = await asyncio.to_thread(_extract_sync)
        except Exception as e:
            logger.error(f"Ontology-guided extraction failed: {e}", exc_info=True)
            return [], []

        relations, nodes_data = _parse_dspy_triplets(triplets)

        # Tidy provenance: tag every relation and node with the matched ontology
        # and the match similarity.
        for _, _, _, props in relations:
            props["ontology"] = ontology_name
            props["ontology_match_similarity"] = round(float(sim), 4)
        for node in nodes_data:
            node["properties"]["ontology"] = ontology_name

        logger.debug(
            "Chunk matched ontology '%s' (sim=%.3f): %d relations, %d nodes.",
            ontology_name, sim, len(relations), len(nodes_data),
        )
        return relations, nodes_data


def get_extractor(config: Dict[str, Any]) -> BaseExtractor:
    """
    Factory function to get the appropriate extractor based on config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured extractor instance
    """
    # Look for backend in extraction settings
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    extractor_type = extraction_config.get('backend', 'dspy') # Default to dspy now

    logger.info(f"Initializing graph extractor: {extractor_type}")

    if extractor_type in ('ontology_dspy', 'ontology'):
        return OntologyGuidedDSPyExtractor(config)

    if extractor_type == 'dspy':
        return DSPyExtractor(config)

    return LangChainExtractor(config)
