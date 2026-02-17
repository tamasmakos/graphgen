import os
import logging
import asyncio
import json
import random
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

import re
from langchain_core.utils.json import parse_json_markdown

from .models import SummarizationTask

logger = logging.getLogger(__name__)

# Configuration
MAX_CONCURRENT_SUMMARIES = 5  # Reduced concurrency due to larger context
CONTEXT_TOKEN_LIMIT = 8000     # Target context size (approx chars * 0.25)
MAX_CHARS_PER_CHUNK = 2000    # Trucate individual chunks to avoid domination
# XML Prompt Definition
COMMUNITY_REPORT_PROMPT = """
<system_role>
You are an expert intelligence analyst specializing in graph-based pattern detection. 
Your goal is to synthesize a comprehensive report on a specific community of entities within a larger network.
</system_role>

<task_description>
Analyze the provided network community structure (entities and relationships) and the associated text chunks.
Identify the core theme, key patterns, and significant insights that define this community.
Produce a structured JSON report that captures the essence of this community.
</task_description>

<input_data_description>
You will be provided with:
1.  **Community Structure**: A list of key entities and their relationships within this group.
2.  **text_chunks**: Raw text segments where these entities appear.
3.  **Sub-community Summaries** (Optional): Summaries of smaller groups contained within this community (for hierarchical context).
</input_data_description>

<output_format>
Return a **SINGLE JSON OBJECT** with the following structure:
{{
    "title": "A short, descriptive title for the community (3-10 words)",
    "summary": "A comprehensive executive summary (3-5 sentences) covering the main topics, entities, and dynamics.",
    "findings": [
        {{
            "summary": "Short title of a key insight/finding",
            "explanation": "Detailed explanation of the finding, citing specific entities or patterns observed."
        }},
        ... (3-5 findings)
    ]
}}
</output_format>

<constraints>
- **Grounding**: Ensure all findings are supported by the provided text chunks or relationships.
- **Completeness**: Prefer synthesis over listing. Connect the dots between entities.
- **Tone**: Professional, analytical, and objective.
- **JSON Only**: Do not output any text outside the JSON block.
- **Strict Syntax**: Ensure the JSON is valid and complete. Double-check for missing commas between array elements and fields.
</constraints>

<context>
{context_xml}
</context>
"""

def _truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text

def _estimate_tokens(text: str) -> int:
    # Rough estimate: 4 chars per token
    return len(text) // 4

def _format_context_xml(task: SummarizationTask) -> str:
    """Format the task data into XML sections for the prompt."""
    xml_parts = []
    current_tokens = 0
    
    # 1. Community Structure (Entities & Relations)
    structure_xml = ["<community_structure>"]
    
    if task.entities:
        structure_xml.append("  <entities>")
        # Sort by degree if available, else random
        sorted_ents = sorted(task.entities, key=lambda x: x.get('degree', 0), reverse=True)[:50]
        for ent in sorted_ents:
            structure_xml.append(f"    <entity name=\"{ent['name']}\" type=\"{ent.get('type', 'Unknown')}\" />")
        structure_xml.append("  </entities>")
        
    if task.relationships:
        structure_xml.append("  <relationships>")
        # Limit relationships
        for src, rel, tgt in task.relationships[:50]: 
            structure_xml.append(f"    <rel source=\"{src}\" type=\"{rel}\" target=\"{tgt}\" />")
        structure_xml.append("  </relationships>")
    
    structure_xml.append("</community_structure>")
    
    structure_str = "\n".join(structure_xml)
    xml_parts.append(structure_str)
    current_tokens += _estimate_tokens(structure_str)
    
    # 2. Sub-community Summaries (if any)
    if task.sub_summaries:
        sub_xml = ["<sub_communities>"]
        for sub in task.sub_summaries:
            sub_xml.append(f"  <sub_community id=\"{sub.get('id')}\">\n    {sub.get('summary')}\n  </sub_community>")
        sub_xml.append("</sub_communities>")
        
        sub_str = "\n".join(sub_xml)
        # Check if we have space
        if current_tokens + _estimate_tokens(sub_str) < CONTEXT_TOKEN_LIMIT:
            xml_parts.append(sub_str)
            current_tokens += _estimate_tokens(sub_str)
        else:
            # Truncate sub-summaries if needed
            logger.warning(f"Truncating sub-summaries for {task.task_id} due to size.")
            # Verify if at least one fits?
            pass

    # 3. Text Chunks (Fill remaining space)
    remaining_tokens = CONTEXT_TOKEN_LIMIT - current_tokens
    if remaining_tokens > 500: # Ensure reasonable space left
        xml_parts.append("<text_chunks>")
        
        chunks_added = 0
        for i, text in enumerate(task.chunk_texts):
            # Dynamic max chars based on remaining space? 
            # Let's keep a hard chunk limit but strictly check total
            clean_text = _truncate_text(text, MAX_CHARS_PER_CHUNK).replace("<", "&lt;").replace(">", "&gt;")
            chunk_str = f"  <chunk id=\"{i}\">\n{clean_text}\n  </chunk>"
            chunk_tokens = _estimate_tokens(chunk_str)
            
            if current_tokens + chunk_tokens < CONTEXT_TOKEN_LIMIT:
                xml_parts.append(chunk_str)
                current_tokens += chunk_tokens
                chunks_added += 1
            else:
                break
                
        xml_parts.append("</text_chunks>")
    
    return "\n".join(xml_parts)

# --- Graph Traversal & Data Collection ---

def get_community_structure(graph: nx.DiGraph, community_nodes: List[str]) -> Tuple[List[Dict], List[Tuple]]:
    """
    Extract entities and internal relationships for a set of nodes in a community.
    community_nodes: List of ENTITY node IDs in this community.
    """
    entities = []
    relationships = []
    
    # Create a set for fast lookup
    comm_node_set = set(community_nodes)
    
    # Get subgraph of these nodes
    subgraph = graph.subgraph(community_nodes)
    
    # Extract Entities
    for node_id in community_nodes:
        if node_id in graph.nodes:
            data = graph.nodes[node_id]
            # Simple degree based on full graph to show global prominence
            degree = graph.degree(node_id)
            entities.append({
                "name": data.get('name', node_id),
                "id": node_id,
                "type": data.get('ontology_class') or data.get('llm_type') or "Entity",
                "degree": degree
            })
            
    # Extract Internal Relationships
    # using subgraph.edges(data=True) gets edges between the nodes in the set
    for u, v, data in subgraph.edges(data=True):
        rel_type = data.get('label') or data.get('relation_type') or "RELATED_TO"
        relationships.append((u, rel_type, v))
        
    return entities, relationships

async def get_chunks_for_community(graph: nx.DiGraph, entity_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Get text chunks connected to these entities.
    Returns (chunk_ids, chunk_texts)
    """
    chunk_ids = set()
    chunk_texts = []
    
    # Find chunks connected to these entities
    # Chunks have HAS_ENTITY edge to Entity (Chunk -> Entity)
    # OR Entity has edge from Chunk (Entity <- Chunk)
    
    # Optimization: Check predecessors of entities
    for ent_id in entity_ids:
        if ent_id in graph:
            for pred in graph.predecessors(ent_id):
                node = graph.nodes[pred]
                # Check for CHUNK type
                if str(node.get('node_type','')).upper() in ['CHUNK', 'TEXTCHUNK']:
                    chunk_ids.add(pred)
    
    # Collect text, sorted by speech_order/chunk_order if available
    sorted_ids = sorted(list(chunk_ids)) # Simple sort for stability
    
    # Better sort: by time/order attributes
    annotated_chunks = []
    for cid in sorted_ids:
        data = graph.nodes[cid]
        order = (data.get('speech_order', 0), data.get('chunk_order', 0))
        annotated_chunks.append((order, cid, data.get('text', '')))
    
    annotated_chunks.sort(key=lambda x: x[0])
    
    final_ids = [x[1] for x in annotated_chunks]
    final_texts = [x[2] for x in annotated_chunks if x[2]]
    
    return final_ids, final_texts

async def collect_task_for_node(graph: nx.DiGraph, node_id: str, is_topic: bool) -> Optional[SummarizationTask]:
    """Build a SummarizationTask for a TOPIC or SUBTOPIC node."""
    node_data = graph.nodes[node_id]
    
    # Parse ID
    # TOPIC_X -> comm_id=X
    # SUBTOPIC_X_Y -> comm_id=X, sub_id=Y
    
    try:
        parts = node_id.split('_')
        community_id = int(parts[1])
        subcommunity_id = int(parts[2]) if not is_topic and len(parts) > 2 else None
    except (IndexError, ValueError):
        logger.error(f"Invalid node format: {node_id}")
        return None

    # Identify member entities
    # For SUBTOPIC: Entity -> IN_TOPIC -> SUBTOPIC (predecessors of subtopic)
    # For TOPIC: Entity -> IN_TOPIC -> TOPIC (predecessors of topic)
    # NOTE: The check 'IN_TOPIC' logic depends on how subcommunities.py built it.
    # It created Entity -> Subtopic and Subtopic -> Parent Topic
    
    member_entities = []
    sub_summaries = []
    
    if is_topic:
        # TOPIC receives edges from SUBTOPICS (Parent -> Child relation reversed in graph? No, usually Child -> Parent for hierarchy)
        # subcommunities.py: graph.add_edge(sub_node_id, topic_node_id, label="PARENT_TOPIC")
        # So Topic predecessors are Subtopics AND stray Entities
        
        for pred in graph.predecessors(node_id):
            p_data = graph.nodes[pred]
            p_type = str(p_data.get('node_type', '')).upper()
            
            if p_type == 'SUBTOPIC':
                # Grab the summary from the subtopic if it exists
                if p_data.get('summary'):
                    sub_summaries.append({
                        "id": pred, 
                        "summary": p_data.get('summary')
                    })
                # We do NOT add subtopic's entities to the Parent's direct entity list 
                # to avoid overwhelming the parent prompt. Parent relies on Sub-summaries.
                # UNLESS the parent has very few subtopics.
                
            elif p_type in ['ENTITY', 'ENTITY_CONCEPT', 'PLACE', 'NAMEDENTITY']:
                member_entities.append(pred)
                
    else:
        # SUBTOPIC predecessors are Entities
        for pred in graph.predecessors(node_id):
            p_data = graph.nodes[pred]
            if str(p_data.get('node_type', '')).upper() in ['ENTITY', 'ENTITY_CONCEPT', 'PLACE', 'NAMEDENTITY']:
                member_entities.append(pred)

    if not member_entities and not sub_summaries:
        logger.warning(f"No content found for {node_id}. Predecessors: {list(graph.predecessors(node_id))}")
        return None
        
    # Collect Structure (Entities & Relations)
    # For Topics, we might want to include "Key Entities" from subtopics? 
    # For now, let's stick to direct members + a sample if purely hierarchical.
    # Actually, if a Topic has NO direct entities (only subtopics), we should probably fetch top entities from subtopics.
    
    if is_topic and not member_entities and sub_summaries:
        # Fetch top entities from subtopics to populate structure
        for sub in sub_summaries:
            sid = sub['id']
            # Get neighbors of subtopic (reverse IN_TOPIC)
            # Predecessors of subtopic are entities
            sub_ents = [p for p in graph.predecessors(sid) if str(graph.nodes[p].get('node_type','')).upper() in ['ENTITY_CONCEPT','ENTITY']]
            member_entities.extend(sub_ents[:5]) # Take top 5 from each
            
    entities_data, relationships_data = get_community_structure(graph, member_entities)
    
    # Collect Chunks
    chunk_ids, chunk_texts = await get_chunks_for_community(graph, member_entities)
    
    return SummarizationTask(
        task_id=node_id,
        community_id=community_id,
        subcommunity_id=subcommunity_id,
        is_topic=is_topic,
        chunk_texts=chunk_texts,
        entities=entities_data,
        relationships=relationships_data,
        sub_summaries=sub_summaries,
        chunk_ids=chunk_ids,
        entity_ids=member_entities
    )

async def process_batch_tasks(llm: Any, tasks: List[SummarizationTask]) -> List[SummarizationTask]:
    """Run a batch of summarization tasks concurrently."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIES)
    
    async def _run(task: SummarizationTask):
        async with semaphore:
            try:
                # Format Prompt
                context_xml = _format_context_xml(task)
                formatted_prompt = COMMUNITY_REPORT_PROMPT.format(context_xml=context_xml)
                
                # Invoke LLM
                response = await llm.ainvoke(formatted_prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Robust parsing
                try:
                    # attempt 1: parse_json_markdown
                    try:
                        data = parse_json_markdown(content)
                    except Exception:
                        # attempt 2: try manual comma fix for common LLM error: } { -> }, {
                        fixed_content = re.sub(r'\}\s*\{', '}, {', content)
                        data = parse_json_markdown(fixed_content)
                    
                    if not isinstance(data, dict):
                         # attempt 3: find { ... } and parse
                         match = re.search(r'\{.*\}', content, re.DOTALL)
                         if match:
                             data = json.loads(match.group(0))
                         else:
                             raise ValueError("No JSON object found in output")
                             
                    task.title = data.get('title', 'Untitled')
                    task.summary = data.get('summary', '')
                    task.findings = data.get('findings', [])
                    
                    # Store full structured report
                    task.structured_report = data.get('findings', [])
                    
                    logger.info(f"Generated report for {task.task_id}: {task.title}")
                    
                except Exception as parse_err:
                    logger.error(f"JSON Parse Error for {task.task_id}: {parse_err}")
                    logger.debug(f"Raw response for {task.task_id}:\n{content}")
                    
                    # Fallback
                    task.title = "Analysis Partially Failed"
                    task.summary = f"Could not parse LLM response. Error: {str(parse_err)}"
                    # Try to regex out the title if possible
                    title_match = re.search(r'"title":\s*"([^"]+)"', content)
                    if title_match:
                        task.title = title_match.group(1)
                    
            except Exception as e:
                logger.error(f"Error processing {task.task_id}: {e}")
                task.title = "Error"
                task.summary = f"Processing error: {str(e)}"
    
    await asyncio.gather(*[_run(t) for t in tasks])
    return tasks

async def update_graph_nodes(graph: nx.DiGraph, tasks: List[SummarizationTask]):
    """Update graph nodes with the generated reports."""
    for task in tasks:
        node = graph.nodes[task.task_id]
        
        # Basic fields
        node['title'] = task.title
        node['summary'] = task.summary
        node['name'] = task.title # Display name
        
        # Store detailed findings as JSON string (Neo4j/GraphML compatibility)
        if task.findings:
            node['findings_json'] = json.dumps(task.findings)
            
        # Metadata
        node['entity_count'] = len(task.entity_ids)
        node['relationship_count'] = len(task.relationships)
        node['chunk_count'] = len(task.chunk_ids)
        node['updated_at'] = datetime.now().isoformat()
        node['has_summary'] = True

# --- Main Pipeline ---

async def generate_community_summaries(graph: nx.DiGraph, llm: Any) -> Dict[str, Any]:
    """
    Main entry point.
    1. Identify Subtopics -> Generate Summaries
    2. Identify Topics -> Generate Summaries (using Subtopic context)
    """
    start_time = datetime.now()
    
    # Identify nodes
    subtopic_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SUBTOPIC']
    topic_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'TOPIC']
    
    logger.info(f"Found {len(subtopic_nodes)} Subtopics and {len(topic_nodes)} Topics.")
    
    # 1. Process Subtopics
    logger.info("Phase 1: Summarizing Subtopics...")
    subtopic_tasks = []
    for node_id in subtopic_nodes:
        task = await collect_task_for_node(graph, node_id, is_topic=False)
        if task:
            subtopic_tasks.append(task)
        else:
            logger.warning(f"Failed to create task for {node_id}")
    
    logger.info(f"Subtopic tasks collected: {len(subtopic_tasks)}")
            
    await process_batch_tasks(llm, subtopic_tasks)
    await update_graph_nodes(graph, subtopic_tasks)
    
    logger.info(f"Phase 1 Complete. {len(subtopic_tasks)} subtopics summarized.")
    
    # 2. Process Topics
    # Now that Subtopics have summaries in the graph, we can collect them for Topics
    logger.info("Phase 2: Summarizing Topics...")
    topic_tasks = []
    for node_id in topic_nodes:
        task = await collect_task_for_node(graph, node_id, is_topic=True)
        if task:
            topic_tasks.append(task)
            
    await process_batch_tasks(llm, topic_tasks)
    await update_graph_nodes(graph, topic_tasks)
    
    logger.info(f"Phase 2 Complete. {len(topic_tasks)} topics summarized.")
    
    return {
        "processing_time": (datetime.now() - start_time).total_seconds(),
        "subtopics_processed": len(subtopic_tasks),
        "topics_processed": len(topic_tasks)
    }
