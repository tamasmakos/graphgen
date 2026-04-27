import asyncio
import functools
import logging
import re
from typing import List, Dict, Any, Tuple, Set, Optional

import networkx as nx
import torch

# --- Graphgen Imports ---
from graphgen.data_types import PipelineContext, ChunkExtractionTask
from graphgen.pipeline.entity_relation.extractors import BaseExtractor, get_extractor
from graphgen.prototype_gliner2_ontology import (
    build_top_level_label_space,
    build_gliner2_schema,
    refine_entities_with_ontology,
    select_candidate_labels,
)
from graphgen.utils.utils import standardize_label
# We assume resolution is in graph_cleaning as previously found
from graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

# --- GLiNER Helper ---
from gliner import GLiNER
from gliner2 import GLiNER2
import spacy

logger = logging.getLogger(__name__)

GLINER_MODEL = None
GLINER2_MODEL = None
SPACY_MODEL = None

def get_gliner2_model(config: Optional[Dict[str, Any]] = None):
    global GLINER2_MODEL
    if GLINER2_MODEL is None:
        try:
            model_name = "fastino/gliner2-base-v1"
            if config:
                extraction_cfg = config.get('extraction', {})
                if hasattr(extraction_cfg, 'model_dump'):
                    extraction_cfg = extraction_cfg.model_dump()
                model_name = extraction_cfg.get('gliner_model', model_name)
            logger.info(f"Loading GLiNER2 model ({model_name})...")
            GLINER2_MODEL = GLiNER2.from_pretrained(model_name)
            logger.info("GLiNER2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER2: {e}")
            return None
    return GLINER2_MODEL

def get_gliner_model(config: Optional[Dict[str, Any]] = None):
    global GLINER_MODEL
    if GLINER_MODEL is None:
        try:
            # Default values
            device = "cpu"
            use_onnx = False
            model_name = "urchade/gliner_medium-v2.1"
            
            if config:
                extraction_cfg = config.get('extraction', {})
                if hasattr(extraction_cfg, 'model_dump'):
                    extraction_cfg = extraction_cfg.model_dump()
                
                requested_device = extraction_cfg.get('device', 'auto')
                if requested_device == 'auto':
                    # Robust check for CUDA
                    try:
                        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                            # Test if we can actually allocate a tensor
                            torch.zeros(1).cuda()
                            device = "cuda"
                        else:
                            device = "cpu"
                    except Exception:
                        device = "cpu"
                else:
                    device = requested_device
                
                use_onnx = extraction_cfg.get('use_onnx', False)
                model_name = extraction_cfg.get('gliner_model', model_name)
            else:
                device = "cpu" # Default to cpu if no config to be safe

            logger.info(f"Loading GLiNER model ({model_name}) on {device} (use_onnx={use_onnx})...")
            
            try:
                # GLiNER 0.2.x uses map_location instead of device for from_pretrained
                GLINER_MODEL = GLiNER.from_pretrained(
                    model_name, 
                    map_location=device,
                    load_onnx_model=use_onnx
                )
            except Exception as e:
                if device == "cuda":
                    logger.warning(f"Failed to load GLiNER on CUDA ({e}), falling back to CPU...")
                    GLINER_MODEL = GLiNER.from_pretrained(
                        model_name, 
                        map_location="cpu",
                        load_onnx_model=use_onnx
                    )
                else:
                    raise e
                    
            logger.info(f"GLiNER model loaded on {device}.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER: {e}")
            return None
    return GLINER_MODEL

def get_spacy_model(model_name: str = "en_core_web_lg"):
    global SPACY_MODEL
    if SPACY_MODEL is None:
        try:
            logger.info(f"Loading Spacy model ({model_name})...")
            SPACY_MODEL = spacy.load(model_name)
            logger.info("Spacy model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Spacy model {model_name}: {e}")
            logger.info(f"Trying to download {model_name}...")
            try:
                from spacy.cli import download
                download(model_name)
                SPACY_MODEL = spacy.load(model_name)
                logger.info("Spacy model loaded after download.")
            except Exception as e2:
                logger.error(f"Failed to download/load Spacy model: {e2}")
                return None
    return SPACY_MODEL

# --- Helper Functions ---

def get_max_concurrent(config: Dict[str, Any], default: int = 8) -> int:
    """Get max_concurrent_extractions from config with fallback to default."""
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
        
    return extraction_cfg.get('max_concurrent_chunks', default)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Simple regex for sentence splitting
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]

# --- Async Extraction ---

async def extract_relations_with_llm_async(
    text: str,
    extractor: BaseExtractor,
    keywords: List[str] = None,
    entities: List[str] = None,
    abstract_concepts: List[str] = None
) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Extract relations using configured extractor."""
    if not extractor or not text:
        return [], []

    try:
        return await extractor.extract_relations(
            text=text,
            keywords=keywords,
            entities=entities,
            abstract_concepts=abstract_concepts
        )
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return [], []


async def _extract_entities_for_chunk(
    text: str, 
    config: Dict[str, Any],
    labels: List[str]
) -> List[Dict[str, Any]]:
    """Run NER (GLiNER, GLiNER2, or Spacy) for a single chunk."""
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()
        
    extraction_backend = extraction_config.get('ner_backend') or extraction_config.get('backend', 'gliner')
    
    if extraction_backend == 'spacy':
        try:
            spacy_model_name = extraction_config.get('spacy_model', 'en_core_web_lg')
            nlp = get_spacy_model(spacy_model_name)
            if nlp:
                # Run spacy in thread to avoid blocking loop
                doc = await asyncio.to_thread(nlp, text)
                return [{
                    "text": standardize_label(ent.text),
                    "label": standardize_label(ent.label_),
                    "ontology_label": standardize_label(ent.label_),
                    "confidence": 1.0,
                    "candidate_labels": labels,
                } for ent in doc.ents]
        except Exception as e:
            logger.error(f"Spacy extraction failed for chunk: {e}")
            return []
    elif extraction_backend == 'gliner2':
        try:
            if not labels:
                logger.warning("No labels provided for GLiNER2 extraction. Skipping inference.")
                return []

            label_space = build_top_level_label_space(labels)
            top_k = extraction_config.get('gliner2_top_k_labels', 5)
            candidate_labels = select_candidate_labels(text, label_space, top_k=top_k)
            schema = build_gliner2_schema(candidate_labels, extraction_config.get('gliner2_label_descriptions'))
            model = get_gliner2_model(config)
            if not model:
                return []

            threshold = extraction_config.get('gliner_threshold', 0.5)
            result = await asyncio.to_thread(
                model.extract,
                text,
                schema,
                threshold,
                True,
                True,
                True,
            )

            flat_entities = []
            for label, spans in (result.get('entities') or {}).items():
                for span in spans:
                    entity = {
                        'text': standardize_label(span.get('text', '')),
                        'label': standardize_label(label),
                        'ontology_label': standardize_label(label),
                        'confidence': float(span.get('confidence', 0.0)),
                        'start': span.get('start'),
                        'end': span.get('end'),
                        'candidate_labels': candidate_labels,
                        'backend': 'gliner2',
                    }
                    flat_entities.append(entity)

            return refine_entities_with_ontology(flat_entities, label_space)
        except Exception as e:
            logger.error(f"GLiNER2 extraction failed for chunk: {e}")
            return []
    else:
        # GLiNER Default
        try:
            model = get_gliner_model(config)
            if model:
                # Split sentences for better GLiNER performance on long text
                sentences = split_sentences(text)
                if not sentences:
                    sentences = [text]
                
                # Read threshold from config (avoid hardcoding)
                threshold = extraction_config.get('gliner_threshold', 0.5)
                
                # Check for empty labels to prevent GLiNER internal max() error
                if not labels:
                    logger.warning("No labels provided for GLiNER extraction. Skipping inference.")
                    return []
                
                # Use the modern inference API (batch_predict_entities is deprecated in GLiNER >= 0.2.x)
                predict_func = functools.partial(
                    model.inference, 
                    sentences, 
                    labels, 
                    threshold=threshold
                )
                
                predictions = await asyncio.to_thread(predict_func)
                flat_predictions = []
                for batch in predictions or []:
                    if isinstance(batch, list):
                        flat_predictions.extend(batch)
                    elif isinstance(batch, dict):
                        flat_predictions.append(batch)
                return [{
                    "text": standardize_label(item.get('text', '')),
                    "label": standardize_label(item.get('label', '')),
                    "ontology_label": standardize_label(item.get('label', '')),
                    "confidence": float(item.get('score', item.get('confidence', 0.0)) or 0.0),
                    "candidate_labels": labels,
                    "backend": 'gliner'
                } for item in flat_predictions]
                
        except Exception as e:
            logger.error(f"GLiNER extraction failed for chunk: {e}")
            return []

    return []

async def process_extraction_task(
    deps: PipelineContext,
    task: ChunkExtractionTask,
    semaphore: asyncio.Semaphore,
    extractor: BaseExtractor,
    config: Dict[str, Any],
    ontology_labels: List[str]
) -> Dict[str, Any]:
    """Process a single chunk extraction task"""
    async with semaphore:
        # Use full chunk_id to avoid confusion with similar short IDs
        logger.debug(f"      Starting {task.chunk_id}")
        
        try:
            # 1. Run NER for this chunk (GLiNER/Spacy)
            # This is now done in parallel with other chunks' NER/LLM steps
            gliner_entities = await _extract_entities_for_chunk(task.chunk_text, config, ontology_labels)
            
            # 2. Prepare allowed nodes/types for LLM (Gatekeeper Logic)
            # Find which ontology labels were actually discovered in this chunk
            found_labels = set()
            for ent in gliner_entities:
                label = ent.get('label')
                if label:
                    found_labels.add(label)
                    
            # THE ONTOLOGY LABELS are filtered to only those found in this chunk
            node_labels = list(found_labels)
            
            if not node_labels:
                 logger.debug(f"Chunk {task.chunk_id}: Found labels: []")
            else:
                 logger.debug(f"Chunk {task.chunk_id}: Found labels: {node_labels}")
            
            # Fallback
            if not node_labels and ontology_labels:
                 node_labels = ontology_labels
                 logger.debug(f"Chunk {task.chunk_id}: No labels found by NER, using full ontology set as fallback")
            
            # THE DISCOVERED ENTITIES are guidance/hints for extraction
            discovered_entities = []
            if task.entities:
                discovered_entities.extend(task.entities)
            if gliner_entities:
                gliner_texts = [e.get('text') for e in gliner_entities if e.get('text')]
                discovered_entities.extend(gliner_texts)
            
            # Deduplicate Discoveries
            discovered_entities = list(set(discovered_entities))
            
            # Run LLM Relation Extraction
            raw_relations, raw_nodes = await extract_relations_with_llm_async(
                text=task.chunk_text,
                extractor=extractor,
                keywords=task.keywords,
                entities=discovered_entities, 
                abstract_concepts=node_labels 
            )
            
            chunk_data = {
                'knowledge_triplets': raw_relations,
                'raw_extraction': {
                    'relations': raw_relations,
                    'nodes': raw_nodes
                },
                'gliner_entities': gliner_entities
            }
            deps.graph.nodes[task.chunk_id].update(chunk_data)
            
            unique_relation_entities = {
                x for tr in raw_relations for x in (tr[0], tr[2]) if x
            }
            unique_gliner_entities = {
                e.get('text') for e in gliner_entities if e.get('text')
            }
            rel_count = len(raw_relations)
            ent_count = len(unique_relation_entities or unique_gliner_entities)
            deps.graph.nodes[task.chunk_id]['extraction_successful'] = bool(raw_relations or gliner_entities)
            logger.debug(f"      Completed {task.chunk_id}: stored {ent_count} entities, {rel_count} relations")
            
            return {
                "success": True, 
                "chunk_id": task.chunk_id,
                "entity_count": ent_count,
                "relation_count": rel_count
            }
            
        except Exception as e:
            logger.error(f"      Failed {task.chunk_id}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "chunk_id": task.chunk_id}

async def extract_all_entities_relations(deps: PipelineContext, config: Dict[str, Any], extractor: BaseExtractor = None) -> Dict[str, Any]:
    """Phase 2: Parallel entity/relation extraction"""
    if not deps.extraction_tasks:
        return {"processed": 0, "successful": 0, "errors": []}
    
    # Deduplicate tasks
    seen_chunk_ids = set()
    unique_tasks = []
    for task in deps.extraction_tasks:
        if task.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(task.chunk_id)
            unique_tasks.append(task)
    
    if len(unique_tasks) < len(deps.extraction_tasks):
        logger.warning(f"Removed {len(deps.extraction_tasks) - len(unique_tasks)} duplicate extraction tasks")
    
    should_close_extractor = False
    if extractor is None:
        extractor = get_extractor(config)
        should_close_extractor = True
        
    logger.info(f"Using {extractor.__class__.__name__} for relation extraction")
    
    # Resolve ontology labels ONCE
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
        
    from graphgen.utils.labels import resolve_entity_labels
    labels = resolve_entity_labels(extraction_cfg)
    logger.info(f"Using {len(labels)} entity labels for extraction pipeline. Processing {len(unique_tasks)} chunks in parallel...")

    # Preload NER model to prevent race conditions in parallel threads
    extraction_backend = extraction_cfg.get('ner_backend') or extraction_cfg.get('backend', 'gliner')
    if extraction_backend == 'spacy':
        spacy_model_name = extraction_cfg.get('spacy_model', 'en_core_web_lg')
        get_spacy_model(spacy_model_name)
    elif extraction_backend == 'gliner2':
        get_gliner2_model(config)
    else:
        get_gliner_model(config)

    try:
        # Use max_concurrent from config for controlled parallelization
        max_concurrent = get_max_concurrent(config, default=8)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = [process_extraction_task(
            deps, 
            task, 
            semaphore, 
            extractor, 
            config,
            labels
        ) for task in unique_tasks]
        
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r.get("success"))
        errors = [r.get("error") for r in results if not r.get("success") and r.get("error")]
        
        logger.info(f"Extraction complete: {successful}/{len(results)} successful")
        
        # Enrich graph per segment
        enrich_result = await enrich_graph_per_segment(deps)
        
        total_entities = sum(r.get("entity_count", 0) for r in results if r.get("success"))
        total_relations = sum(r.get("relation_count", 0) for r in results if r.get("success"))

        return {
            "processed": len(results),
            "successful": successful,
            "total_entities_extracted": total_entities,
            "total_relations_extracted": total_relations,
            "errors": errors + enrich_result.get("errors", [])
        }
    finally:
        if should_close_extractor and hasattr(extractor, 'close'):
            await extractor.close()

# --- Graph Enrichment ---


def _get_chunks_for_segment(graph: nx.DiGraph, segment_id: str) -> List[str]:
    """Find all chunks associated with a segment, including via Conversations and Contexts."""
    chunk_ids = set()
    
    # 1. Direct Chunks (old schema)
    for neighbor in graph.neighbors(segment_id):
        edge = graph.get_edge_data(segment_id, neighbor) or {}
        # Relaxed check: Accept TextChunk or CHUNK, or just trust the edge
        neighbor_type = graph.nodes[neighbor].get('node_type', '')
        if edge.get('label') == 'HAS_CHUNK' and neighbor_type in ['CHUNK', 'TextChunk', 'Chunk']:
            chunk_ids.add(neighbor)
            
    # 2. Via Conversations
    # Segment -> HAS_CONVERSATION -> Conversation
    for neighbor in graph.neighbors(segment_id):
        edge = graph.get_edge_data(segment_id, neighbor) or {}
        if edge.get('label') == 'HAS_CONVERSATION':
            conv_id = neighbor
            # Conversation -> HAS_CHUNK -> Chunk
            for conv_neighbor in graph.neighbors(conv_id):
                conv_edge = graph.get_edge_data(conv_id, conv_neighbor) or {}
                if conv_edge.get('label') == 'HAS_CHUNK' and graph.nodes[conv_neighbor].get('node_type') == 'CHUNK':
                    chunk_ids.add(conv_neighbor)
                
                # Conversation -> HAS_CONTEXT -> Context -> HAS_DESCRIPTION_CHUNK -> Chunk
                if conv_edge.get('label') == 'HAS_CONTEXT':
                    ctx_id = conv_neighbor
                    for ctx_neighbor in graph.neighbors(ctx_id):
                        ctx_edge = graph.get_edge_data(ctx_id, ctx_neighbor) or {}
                        if ctx_edge.get('label') == 'HAS_DESCRIPTION_CHUNK' and graph.nodes[ctx_neighbor].get('node_type') == 'CHUNK':
                            chunk_ids.add(ctx_neighbor)

    return list(chunk_ids)

async def add_triplets_to_graph_for_segment(
    deps: PipelineContext,
    relations: List[Tuple[str, str, str, Dict[str, Any]]],
    entity_mappings: Dict[str, str],
    segment_id: str,
    chunk_entity_map: Dict[str, Set[str]],
    gliner_label_map: Dict[str, str] = None,  # New parameter
    llm_type_map: Dict[str, str] = None # Map of entity -> llm_type
):
    """Write entities/edges to graph for a segment"""
    graph = deps.graph
    gliner_label_map = gliner_label_map or {}
    llm_type_map = llm_type_map or {}
    
    # Add entities
    for _, mapped_ent in entity_mappings.items():
        # Classify entity using LLM type map or GLiNER map
        # LLM type is highly reliable now as it follows ontology strict_mode
        llm_type = llm_type_map.get(mapped_ent)
        gliner_label_entry = gliner_label_map.get(mapped_ent)
        if not gliner_label_entry:
            gliner_label_entry = gliner_label_map.get(mapped_ent.lower())
        gliner_label = gliner_label_entry.get('label') if isinstance(gliner_label_entry, dict) else gliner_label_entry
            
        # Use LLM type if available, fallback to GLiNER/Spacy, then default
        ontology_label = llm_type or gliner_label or "Concept"
            
        if not graph.has_node(mapped_ent):
            # Create new node
            graph.add_node(mapped_ent, 
                         node_type="ENTITY_CONCEPT", 
                         ontology_class=ontology_label,
                         llm_type=llm_type,
                         name=mapped_ent, 
                         graph_type="entity_relation")
        else:
            # Update existing node
            node_data = graph.nodes[mapped_ent]
            
            # Ensure name is set
            if 'name' not in node_data:
                node_data['name'] = mapped_ent
                
            # Update llm_type if available
            if llm_type:
                node_data['llm_type'] = llm_type
                
            # Smart update for ontology_class: prioritize LLM type, then GLiNER
            if llm_type:
                node_data['ontology_class'] = llm_type
            elif not node_data.get('ontology_class') or node_data.get('ontology_class') == 'Concept':
                if gliner_label:
                    node_data['ontology_class'] = gliner_label
                elif 'ontology_class' not in node_data:
                    node_data['ontology_class'] = 'Concept'
                
            # Ensure node_type is set (if it was created implicitly)
            if 'node_type' not in node_data:
                node_data['node_type'] = "ENTITY_CONCEPT"
                node_data['graph_type'] = "entity_relation"
 
    
    # Add relations
    for h, r, t, props in relations:
        if not graph.has_node(h) or not graph.has_node(t):
            continue
            
        # Props usually contains confidence, evidence
        edge_attrs = {
            "label": r, 
            "relation_type": r, 
            "graph_type": "entity_relation", 
            "segment_id": segment_id, 
            "source": "extraction"
        }
        if props:
            edge_attrs.update(props)
            
        graph.add_edge(h, t, **edge_attrs)

    # Link chunks to entities
    for chunk_id, entities in chunk_entity_map.items():
        for ent in entities:
            mapped = entity_mappings.get(ent, ent)
            if graph.has_node(mapped):
                graph.add_edge(chunk_id, mapped, label="HAS_ENTITY", graph_type="lexical_graph")

async def enrich_graph_per_segment(deps: PipelineContext) -> Dict[str, Any]:
    """Aggregate chunk-level extractions per segment, run coref, and enrich graph."""
    graph = deps.graph
    segment_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SEGMENT']
    
    # Fallback to Document nodes if no SEGMENT nodes found (e.g. simplified schema)
    if not segment_nodes:
        segment_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') in ['Document', 'DAY', 'Doc', 'DOC']]
        if segment_nodes:
            logger.info(f"Enriching graph using {len(segment_nodes)} Document nodes (no SEGMENT nodes found)")

    segments_processed = 0
    errors = []
    
    for segment_id in segment_nodes:
        try:
            chunk_ids = _get_chunks_for_segment(graph, segment_id)
            if not chunk_ids:
                continue
            
            aggregated_relations = []
            chunk_entity_map = {}
            all_entities = set()
            
            # Collect GLiNER labels from all chunks
            gliner_label_map = {}
            
            for cid in chunk_ids:
                node = graph.nodes.get(cid, {})
                raw_extraction = node.get('raw_extraction') or {}
                raw_relations = raw_extraction.get('relations') or []
                raw_nodes = raw_extraction.get('nodes') or []
                
                gliner_entities = node.get('gliner_entities', [])
                
                # Populate GLiNER map
                for entity in gliner_entities:
                    text = standardize_label(entity.get('text', ''))
                    ontology_label = standardize_label(entity.get('ontology_label') or entity.get('label', ''))
                    if text and ontology_label:
                        current = gliner_label_map.get(text)
                        current_conf = 0.0 if not isinstance(current, dict) else float(current.get('confidence', 0.0) or 0.0)
                        new_conf = float(entity.get('confidence', 0.0) or 0.0)
                        if current is None or new_conf >= current_conf:
                            gliner_label_map[text] = {
                                'label': ontology_label,
                                'confidence': new_conf,
                                'backend': entity.get('backend'),
                                'start': entity.get('start'),
                                'end': entity.get('end'),
                                'candidate_labels': entity.get('candidate_labels', []),
                            }
                
                entities_in_chunk = set()
                for (h, r, t, _) in raw_relations:
                    aggregated_relations.append((h, r, t, _))
                    entities_in_chunk.add(h)
                    entities_in_chunk.add(t)
                
                # Also add entities found by Spacy/GLiNER even if no relations found
                if gliner_entities:
                     for entity in gliner_entities:
                        text = entity.get('text', '').strip()
                        if text:
                            entities_in_chunk.add(text)
                
                if not entities_in_chunk:
                    initial_ents = node.get('initial_entities') or []
                    entities_in_chunk.update([standardize_label(e) for e in initial_ents])
                
                if entities_in_chunk:
                    chunk_entity_map[cid] = entities_in_chunk
                    all_entities.update(entities_in_chunk)
            
            if not aggregated_relations and not all_entities:
                continue

            coref_result = resolve_extraction_coreferences(aggregated_relations, list(all_entities))
            cleaned_relations = coref_result.get('cleaned_relations', [])
            entity_mappings = coref_result.get('entity_mappings', {})
            
            # Build LLM type map
            llm_type_map = {}
            
            # Re-iterating to be safe and simple
            for cid in chunk_ids:
                node = graph.nodes.get(cid, {})
                raw_nodes = (node.get('raw_extraction') or {}).get('nodes') or []
                for n in raw_nodes:
                    ent_id = n.get('id')
                    ent_type = n.get('type')
                    if ent_id and ent_type:
                        llm_type_map[ent_id] = ent_type
                        
            canonical_llm_type_map = {}
            for ent_id, ent_type in llm_type_map.items():
                # ent_id is the raw extracted name
                canonical = entity_mappings.get(ent_id, ent_id)
                if canonical:
                    canonical_llm_type_map[canonical] = ent_type

            await add_triplets_to_graph_for_segment(
                deps=deps,
                relations=cleaned_relations,
                entity_mappings=entity_mappings,
                segment_id=segment_id,
                chunk_entity_map=chunk_entity_map,
                gliner_label_map=gliner_label_map, # Pass the map
                llm_type_map=canonical_llm_type_map
            )
            
            segments_processed += 1
        except Exception as e:
            logger.warning(f"Segment-level enrichment failed for {segment_id}: {e}")
            errors.append(f"{segment_id}: {e}")
            continue
    
    return {"segments_processed": segments_processed, "errors": errors}
