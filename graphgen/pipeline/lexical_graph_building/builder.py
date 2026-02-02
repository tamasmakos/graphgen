import os
import asyncio
import logging
from datetime import datetime, date
from typing import Dict, Any, List
import networkx as nx

from graphgen.types import PipelineContext, ChunkExtractionTask, SegmentData
from graphgen.pipeline.lexical_graph_building.chunking import process_document_splitting
from graphgen.utils.parsers.life import LifeLogParser
from graphgen.config.schema import GraphSchema

logger = logging.getLogger(__name__)

def get_max_concurrent(config: Dict[str, Any], default: int = 8) -> int:
    """Get max_concurrent_extractions from config with fallback to default."""
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
        
    return extraction_cfg.get('max_concurrent_chunks', default)

async def process_single_segment(
    deps: PipelineContext,
    segment: SegmentData,
    doc_id: str,
    segment_index: int,
    global_segment_order: int,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    schema: GraphSchema = None
) -> Dict[str, Any]:
    async with semaphore:
        chunk_count = 0
        
        # Determine Node Labels from Schema
        chunk_label = "CHUNK" # Default
        if schema and "Chunk" in schema.nodes:
            chunk_label = schema.nodes["Chunk"].label
            
            # If "Chunk" source type is "segment", then the Segment IS the Chunk
            # But here we have logic: Document -> Segment -> Chunk (via splitting)
            # The User wants Doc -> Chunk.
            # So if we map Segment -> Chunk directly, we skip splitting?
            # Or does "Chunk" imply the result of splitting?
            # For now, let's assume the Schema "Chunk" corresponds to the split outputs 
            # OR if source_type="chunk", it's the leaf node.
            
        segment_label = "SEGMENT"
        if schema and "Segment" in schema.nodes:
             segment_label = schema.nodes["Segment"].label
        
        # Set name field: first 20 chars of content
        segment_name = segment.content[:20].strip() if segment.content else f"Segment {segment_index}"
        if len(segment.content) > 20:
            segment_name += "..."
        
        # If schema simplifies Doc -> Chunk, we might want to skip the intermediate SEGMENT node
        # But for now, let's keep the structure but rename if needed.
        # If schema doesn't define "Segment", maybe we treat this node strictly as a holder?
        
        # Let's map Segment -> Chunk if that's what's requested?
        # The user said: "The Doc, Chunk, Entity, Topic and Subtopic Nodes must not change."
        # This implies NO "Segment" node in the final graph?
        # If so, we should skip adding the SEGMENT node if it's not in schema?
        # But we need a parent for the chunks?
        
        # Heuristic: If schema has "Chunk" but not "Segment", we might treat the split chunks as the direct children of Doc.
        # Let's preserve the existing logic but use schema labels where they match.
        
        # For this specifc request, "Chunk" seems to be the leaf.
        # Let's assume we proceed to splitting.
        
        # Split text (Lexical Chunking)
        try:
            chunks = await process_document_splitting(segment.content, config)
            
            for i, chunk_text in enumerate(chunks):
                # ID generation
                chunk_id = f"{doc_id}_C{segment_index}_{i}"
                
                chunk_name = chunk_text[:20].strip() if chunk_text else f"Chunk {i}"
                if len(chunk_text) > 20:
                    chunk_name += "..."
                
                deps.graph.add_node(chunk_id,
                                  node_type=chunk_label,
                                  graph_type="lexical_graph",
                                  text=chunk_text,
                                  length=len(chunk_text),
                                  initial_entities=[],
                                  llama_metadata={}, 
                                  name=chunk_name,
                                  # Add attributes from segment metadata directly to Chunk if skipping segment node
                                  **segment.metadata)
                
                # Edge: DOC -> CHUNK (Directly, skipping Segment if not needed?)
                # If schema says Doc -> Chunk, we link Doc -> Chunk
                deps.graph.add_edge(doc_id, chunk_id, label="HAS_CHUNK", graph_type="lexical_graph")
                
                deps.extraction_tasks.append(ChunkExtractionTask(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    entities=[],
                    abstract_concepts=[], 
                    keywords=[]
                ))
                chunk_count += 1
                
        except Exception as e:
            logger.error(f"Error processing segment {segment.segment_id} with splitting: {e}", exc_info=True)
        
        return {"chunk_count": chunk_count, "segment_id": segment.segment_id}

async def add_segments_to_graph(deps: PipelineContext, segments: List[SegmentData], doc_id: str, config: Dict[str, Any] = None, schema: GraphSchema = None) -> Dict[str, Any]:
    config = config or {}
    chunk_count = 0
    segment_count = 0
    
    # Get max_concurrent from config for controlled parallelization
    max_concurrent = get_max_concurrent(config, default=8)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process segments concurrently
    tasks = []
    
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    limit = extraction_config.get('speech_limit', 0)
    # Also support old key if needed, or just rely on 'speech_limit'
    if 'segment_limit' in config:
         limit = config['segment_limit']
    
    for idx, segment in enumerate(segments):
        # Check limit before creating task
        if limit > 0 and deps.total_segments >= limit:
            break
        
        # Capture current global_segment_order before incrementing
        current_global_order = deps.total_segments
            
        task = process_single_segment(
            deps=deps,
            segment=segment,
            doc_id=doc_id,
            segment_index=idx,
            global_segment_order=current_global_order,
            config=config,
            semaphore=semaphore,
            schema=schema
        )
        tasks.append(task)
        segment_count += 1
        deps.total_segments += 1
        
        # Check limit after incrementing
        if limit > 0 and deps.total_segments >= limit:
            break
    
    # Wait for all segment processing tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    errors = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
        else:
            chunk_count += result.get("chunk_count", 0)
    
    return {"segments_count": segment_count, "chunks_count": chunk_count, "errors": errors}

async def process_single_document_lexical(deps: PipelineContext, filename: str, input_dir: str, config: Dict[str, Any] = None, schema: GraphSchema = None, parser: Any = None) -> Dict[str, Any]:
    """Process a single document."""
    config = config or {}
    
    # Use provided parser or default
    if not parser:
        if filename.lower().endswith('.csv'):
             parser = LifeLogParser()
        else:
             from graphgen.utils.parsers.custom import RegexParser
             # Use a regex that never matches to treat whole file as one segment
             parser = RegexParser(segment_splitter="(?!)", attributes_map={})
    
    # Determine Node Labels from Schema
    doc_label = "DAY" 
    if schema and "Doc" in schema.nodes:
        doc_label = schema.nodes["Doc"].label
        
    try:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading {filename}: {str(e)}")
        return {"error": f"Error reading {filename}: {str(e)}", "segments_added": 0, "chunks_added": 0}

    # Try to extract date from filename first
    doc_date_str = parser.extract_date(filename)
    
    # Fallback to content-based extraction
    if not doc_date_str:
        doc_date_str = parser.extract_date_from_content(text)
        
    if not doc_date_str:
        logger.warning(f"Could not extract date from {filename}, using today's date")
        doc_date_str = datetime.now().strftime("%Y-%m-%d")
        
    doc_date_obj = datetime.strptime(doc_date_str, "%Y-%m-%d").date()
    
    # Create Document Node ID
    # Use filename as ID base if possible for uniformity, else use date (legacy)
    if doc_label == "Doc":
        # Matches user request
        doc_id = f"DOC_{filename}"
    else:
        doc_id = f"{doc_label}_{doc_date_str}"
    
    try:
        logger.info(f"Processing document {filename} ({len(text)} chars)")
        
        # Add Document/Day node if not exists
        if not deps.graph.has_node(doc_id):
            deps.graph.add_node(doc_id, 
                               node_type=doc_label, 
                               graph_type="lexical_graph",
                               date=doc_date_str,
                               name=filename, # Name should appear as filename for Doc
                               segment_count=0)
        
        # Parse segments
        segments = parser.parse(text, filename, doc_date_obj)
        
        segments_result = await add_segments_to_graph(deps, segments, doc_id, config, schema=schema)
        
        # Update segment count
        deps.graph.nodes[doc_id]['segment_count'] = (deps.graph.nodes[doc_id].get('segment_count', 0) + 
                                                   segments_result.get("segments_count", 0))
        
        return {
            "day_id": doc_id,
            "segments_added": segments_result.get("segments_count", 0),
            "chunks_added": segments_result.get("chunks_count", 0),
            "errors": segments_result.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return {"error": f"Error processing {filename}: {str(e)}", "segments_added": 0, "chunks_added": 0}

async def build_lexical_graph(deps: PipelineContext, input_dir: str, config: Dict[str, Any] = None, schema: GraphSchema = None) -> Dict[str, Any]:
    """Phase 1: Build the complete lexical graph structure sequentially"""
    config = config or {}
    results = {"documents_processed": 0, "total_segments": 0, "total_chunks": 0, "errors": []}
    
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} not found")
        
        # Check if a specific file pattern is provided (for incremental processing)
        extraction_config = config.get('extraction', {})
        if hasattr(extraction_config, 'model_dump'):
            extraction_config = extraction_config.model_dump()
            
        file_pattern = extraction_config.get('file_pattern')
        if file_pattern:
            # Process only the specified file
            import fnmatch
            all_files = os.listdir(input_dir)
            filenames = [f for f in all_files if fnmatch.fnmatch(f, file_pattern)]
            logger.info(f"Processing specific file(s) matching pattern '{file_pattern}': {filenames}")
        else:
            # Process all .txt files
            filenames = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
            logger.info(f"Found {len(filenames)} files to process")
        
        for filename in filenames:
            limit = config.get('segment_limit', config.get('speech_limit', 0))
            if limit > 0 and deps.total_segments >= limit:
                break
                
            doc_result = await process_single_document_lexical(deps, filename, input_dir, config, schema=schema)
                
            results["documents_processed"] += 1
            results["total_segments"] += doc_result.get("segments_added", 0)
            results["total_chunks"] += doc_result.get("chunks_added", 0)
            
            if doc_result.get("errors"):
                results["errors"].extend(doc_result["errors"])
                
        return results
        
    except Exception as e:
        error_msg = f"Error in build_lexical_graph: {str(e)}"
        logger.error(error_msg, exc_info=True)
        results["errors"].append(error_msg)
        return results
