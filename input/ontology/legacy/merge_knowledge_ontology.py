#!/usr/bin/env python3
"""
Knowledge Graph to Ontology Merger

This script efficiently merges entities/concepts from a knowledge graph 
with classes from an ontology using LLM-based semantic matching.

Key features:
- Intelligent batching to minimize LLM API calls
- Smart filtering and pre-processing  
- Rate limiting and retry logic from kg.py
- Outputs merged graph with EXAMPLE_OF relationships
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not LLM_AVAILABLE:
    logger.error("LangChain dependencies required! Install with: pip install langchain langchain-experimental langchain-groq")
    exit(1)

@dataclass
class EntityConcept:
    """Represents an entity or concept from the knowledge graph"""
    name: str
    type: str  # 'entity' or 'concept' 
    source_chunk_id: str
    context: str = ""
    
@dataclass
class OntologyClass:
    """Represents a class from the ontology"""
    uri: str
    label: str
    comment: str
    type: str
    ontology_type: str

@dataclass
class BatchResult:
    """Result from a batch LLM processing"""
    mappings: List[Tuple[str, str]]  # (entity_name, ontology_class_uri)
    successful: bool
    error: Optional[str] = None

class KnowledgeOntologyMerger:
    """Main class for merging knowledge graph with ontology"""
    
    def __init__(self, knowledge_graph_path: str, ontology_path: str):
        self.knowledge_graph_path = knowledge_graph_path
        self.ontology_path = ontology_path
        self.knowledge_graph = None
        self.ontology = None
        self.entities_concepts: List[EntityConcept] = []
        self.ontology_classes: Dict[str, OntologyClass] = {}
        self.merged_graph = None
        
        # Initialize LLM with same settings as kg.py
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4000)),
            top_p=float(os.getenv("LLM_TOP_P", 0.9)),
            presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", 0.3)),
            frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", 0.3))
        )
        
        # Rate limiting settings (copied from kg.py)
        self.max_batch_size = 10  # Much smaller batches to avoid rate limits
        self.max_concurrent_batches = 3  # Reduced from 4 to match kg.py patterns
        
    def load_graphs(self) -> bool:
        """Load both JSON graphs"""
        logger.info("Loading knowledge graph and ontology...")
        
        try:
            # Load knowledge graph
            with open(self.knowledge_graph_path, 'r', encoding='utf-8') as f:
                self.knowledge_graph = json.load(f)
            logger.info(f"Loaded knowledge graph with {len(self.knowledge_graph.get('nodes', []))} nodes")
            
            # Load ontology
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                self.ontology = json.load(f)
            logger.info(f"Loaded ontology with {len(self.ontology.get('classes', {}))} classes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading graphs: {e}")
            return False
    
    def extract_entities_concepts(self) -> None:
        """Extract entities and concepts from entity_relation subgraph nodes only"""
        logger.info("Extracting entities and concepts from entity_relation subgraph...")
        
        # Focus ONLY on entity_relation subgraph nodes with node_type ENTITY_CONCEPT
        for node in self.knowledge_graph.get('nodes', []):
            if (node.get('node_type') == 'ENTITY_CONCEPT' and 
                node.get('graph_type') == 'entity_relation'):
                
                entity_name = node.get('name', '').strip()
                entity_type = node.get('entity_type', 'UNKNOWN').lower()
                source_chunks = node.get('extracted_from', [])
                
                if entity_name and len(entity_name) > 1:
                    # Determine if this is an entity or concept based on entity_type
                    if entity_type in ['person', 'organization', 'location', 'geopolitical_entity', 'event']:
                        concept_type = 'entity'
                    else:
                        concept_type = 'concept'
                    
                    # Use the first source chunk if available
                    source_chunk = source_chunks[0] if source_chunks else node.get('id', '')
                    
                    self.entities_concepts.append(EntityConcept(
                        name=entity_name,
                        type=concept_type,
                        source_chunk_id=source_chunk
                    ))
        
        logger.info(f"Extracted {len([e for e in self.entities_concepts if e.type == 'entity'])} unique entities")
        logger.info(f"Extracted {len([e for e in self.entities_concepts if e.type == 'concept'])} unique concepts")
    
    def extract_ontology_classes(self) -> None:
        """Extract and process ontology classes"""
        logger.info("Processing ontology classes...")
        
        classes = self.ontology.get('classes', {})
        
        for uri, class_data in classes.items():
            label = class_data.get('label', '')
            comment = class_data.get('comment', '') or ''
            class_type = class_data.get('type', 'owl:Class')
            ontology_type = class_data.get('ontology_type', 'unknown')
            
            # Skip classes without meaningful labels
            if not label or len(label.strip()) < 2:
                continue
                
            self.ontology_classes[uri] = OntologyClass(
                uri=uri,
                label=label.strip(),
                comment=comment.strip(),
                type=class_type,
                ontology_type=ontology_type
            )
        
        logger.info(f"Processed {len(self.ontology_classes)} ontology classes")
    
    def create_intelligent_batches(self) -> List[List[EntityConcept]]:
        """Create intelligent batches for efficient LLM processing"""
        logger.info("Creating intelligent batches for LLM processing...")
        
        # Group by type and similarity for better batch coherence
        entities = [e for e in self.entities_concepts if e.type == 'entity']
        concepts = [e for e in self.entities_concepts if e.type == 'concept']
        
        batches = []
        
        # Create mixed batches but group similar types together
        def create_batches_from_list(items: List[EntityConcept], batch_size: int):
            batch_list = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_list.append(batch)
            return batch_list
        
        # Process entities first (usually more specific)
        entity_batches = create_batches_from_list(entities, self.max_batch_size)
        
        # Process concepts (usually more abstract)
        concept_batches = create_batches_from_list(concepts, self.max_batch_size)
        
        batches.extend(entity_batches)
        batches.extend(concept_batches)
        
        logger.info(f"Created {len(batches)} batches for processing")
        logger.info(f"Entity batches: {len(entity_batches)}, Concept batches: {len(concept_batches)}")
        logger.info(f"Estimated API calls needed: {len(batches)} (reduced batch size for rate limiting)")
        
        return batches
    
    def create_classification_prompt(self, batch: List[EntityConcept], ontology_classes: List[OntologyClass]) -> str:
        """Create a specialized prompt for entity-to-class classification"""
        
        # Prepare entity/concept list
        items_text = "\n".join([f"- {item.name} ({item.type})" for item in batch])
        
        # Prepare ontology classes (limit to most relevant)
        classes_text = "\n".join([
            f"- {cls.uri}: {cls.label}" + (f" - {cls.comment[:100]}..." if cls.comment else "")
            for cls in ontology_classes[:100]  # Limit to avoid token overflow
        ])
        
        prompt = f"""You are an expert knowledge graph classifier. Your task is to match entities and concepts from parliamentary debates to appropriate ontology classes.

ENTITIES AND CONCEPTS TO CLASSIFY:
{items_text}

AVAILABLE ONTOLOGY CLASSES:
{classes_text}

INSTRUCTIONS:
1. For each entity/concept, determine if it can be reasonably classified as an EXAMPLE_OF any ontology class
2. Only create mappings where there is a clear semantic relationship
3. Prefer more specific classes over general ones when multiple matches are possible
4. If no suitable class exists, do not force a mapping

OUTPUT FORMAT:
For each valid mapping, output exactly one line in this format:
MAPPING: entity_name -> ontology_class_uri

Example:
MAPPING: European Parliament -> http://publications.europa.eu/ontology/cdm#parliament
MAPPING: resolution -> http://publications.europa.eu/ontology/cdm#resolution

Only output MAPPING lines for clear, confident matches. Do not explain or add extra text."""

        return prompt
    
    async def process_batch_with_llm(self, batch: List[EntityConcept]) -> BatchResult:
        """Process a batch of entities using LLM with retry logic from kg.py"""
        batch_items = [item.name for item in batch]
        logger.info(f"Processing batch with {len(batch)} items: {', '.join(batch_items[:3])}{'...' if len(batch) > 3 else ''}")
        
        # Retry settings (copied from kg.py)
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get relevant ontology classes (all of them for now, could be optimized)
                relevant_classes = list(self.ontology_classes.values())
                
                # Create prompt
                prompt_text = self.create_classification_prompt(batch, relevant_classes)
                
                # Use LLM directly with same approach as kg.py
                response = await self.llm.ainvoke(prompt_text)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parse response to extract mappings
                mappings = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith('MAPPING:'):
                        try:
                            # Parse "MAPPING: entity_name -> ontology_class_uri"
                            parts = line.replace('MAPPING:', '').strip().split(' -> ')
                            if len(parts) == 2:
                                entity_name = parts[0].strip()
                                class_uri = parts[1].strip()
                                
                                # Validate that the entity is in our batch and class exists
                                if any(item.name == entity_name for item in batch) and class_uri in self.ontology_classes:
                                    mappings.append((entity_name, class_uri))
                        except Exception as e:
                            logger.warning(f"Could not parse mapping line: {line} - {e}")
                
                logger.info(f"Batch completed: {len(mappings)} mappings found")
                
                # Log the actual mappings for quality monitoring
                if mappings:
                    logger.info("  Mappings found:")
                    for entity_name, class_uri in mappings:
                        class_label = self.ontology_classes[class_uri].label
                        ontology_type = self.ontology_classes[class_uri].ontology_type
                        logger.info(f"    • {entity_name} → {class_label} ({ontology_type})")
                else:
                    logger.info("  No mappings found for this batch")
                
                return BatchResult(mappings=mappings, successful=True)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(f"      Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff like kg.py
                    continue
                else:
                    logger.error(f"      Failed after {max_retries} attempts: {str(e)}", exc_info=True)
                    return BatchResult(mappings=[], successful=False, error=str(e))
        
        return BatchResult(mappings=[], successful=False, error="Max retries exceeded")
    
    async def process_all_batches(self, batches: List[List[EntityConcept]]) -> List[Tuple[str, str]]:
        """Process all batches with concurrency control (copied from kg.py pattern)"""
        logger.info(f"Processing {len(batches)} batches with max {self.max_concurrent_batches} concurrent...")
        
        # Create semaphore for concurrency control (same as kg.py)
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_single_batch(batch):
            async with semaphore:
                return await self.process_batch_with_llm(batch)
        
        # Process batches
        tasks = [process_single_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all mappings
        all_mappings = []
        successful_batches = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed with exception: {result}")
            elif isinstance(result, BatchResult):
                if result.successful:
                    all_mappings.extend(result.mappings)
                    successful_batches += 1
                else:
                    logger.error(f"Batch {i} failed: {result.error}")
            else:
                logger.error(f"Batch {i} returned unexpected result: {type(result)}")
        
        logger.info(f"Completed processing: {successful_batches}/{len(batches)} batches successful")
        logger.info(f"Total mappings found: {len(all_mappings)}")
        
        return all_mappings
    
    def create_merged_graph(self, mappings: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Create merged graph with new EXAMPLE_OF edges and ontological relationships"""
        logger.info("Creating merged graph with EXAMPLE_OF relationships and ontological structure...")
        
        # Start with the original knowledge graph
        self.merged_graph = json.loads(json.dumps(self.knowledge_graph))  # Deep copy
        
        # Track which ontology classes are actually used (referenced in mappings or have relationships)
        used_ontology_classes = set()
        
        # Add classes referenced in mappings
        for entity_name, class_uri in mappings:
            used_ontology_classes.add(class_uri)
        
        # Add classes that have relationships with used classes
        def add_related_classes(class_uri: str, depth: int = 0):
            """Recursively add related classes to ensure connected ontological structure"""
            if depth > 2 or class_uri in used_ontology_classes:  # Limit depth to avoid explosion
                return
                
            if class_uri in self.ontology_classes:
                used_ontology_classes.add(class_uri)
                class_data = self.ontology.get('classes', {}).get(class_uri, {})
                
                # Add superclasses and subclasses
                for related_uri in class_data.get('subclass_of', []) + class_data.get('superclass_of', []):
                    if related_uri in self.ontology_classes:
                        add_related_classes(related_uri, depth + 1)
        
        # Build connected ontological subgraph
        for entity_name, class_uri in mappings:
            add_related_classes(class_uri)
        
        logger.info(f"Including {len(used_ontology_classes)} ontology classes in merged graph")
        
        # Add ontology classes as new nodes
        for class_uri in used_ontology_classes:
            if class_uri in self.ontology_classes:
                class_data = self.ontology_classes[class_uri]
                ontology_node = {
                    "node_type": "ONTOLOGY_CLASS", 
                    "graph_type": "ontology_graph",
                    "uri": class_uri,
                    "label": class_data.label,
                    "comment": class_data.comment,
                    "ontology_type": class_data.ontology_type,
                    "class_type": class_data.type,
                    "id": f"ONTO_{class_uri.split('#')[-1].split('/')[-1]}"
                }
                self.merged_graph["nodes"].append(ontology_node)
        
        # Add EXAMPLE_OF edges and ontological relationships
        if "links" not in self.merged_graph:
            self.merged_graph["links"] = []
        
        # Create mapping from entity name to node IDs (from entity_relation subgraph)
        entity_to_node_ids = defaultdict(list)
        for node in self.merged_graph["nodes"]:
            if (node.get("node_type") == "ENTITY_CONCEPT" and 
                node.get("graph_type") == "entity_relation"):
                entity_name = node.get("name", "")
                if entity_name:
                    entity_to_node_ids[entity_name].append(node.get("id"))
        
        # Add EXAMPLE_OF edges for entity-to-class mappings
        example_of_edges_added = 0
        for entity_name, class_uri in mappings:
            # Find the ontology class node ID
            onto_node_id = f"ONTO_{class_uri.split('#')[-1].split('/')[-1]}"
            
            # Find all entity nodes with this name
            for entity_node_id in entity_to_node_ids.get(entity_name, []):
                edge = {
                    "source": entity_node_id,
                    "target": onto_node_id,
                    "label": "EXAMPLE_OF",
                    "graph_type": "ontology_mapping",
                    "relationship_type": "EXAMPLE_OF",
                    "entity_name": entity_name,
                    "ontology_class_uri": class_uri,
                    "mapping_confidence": "llm_generated"
                }
                self.merged_graph["links"].append(edge)
                example_of_edges_added += 1
        
        # Add ontological relationships between classes
        ontological_edges_added = 0
        
        # Create class URI to node ID mapping for ontology classes
        class_uri_to_node_id = {}
        for class_uri in used_ontology_classes:
            class_uri_to_node_id[class_uri] = f"ONTO_{class_uri.split('#')[-1].split('/')[-1]}"
        
        # 1. Add subclass relationships (rdfs:subClassOf)
        for class_uri in used_ontology_classes:
            if class_uri in self.ontology.get('classes', {}):
                class_data = self.ontology['classes'][class_uri]
                source_node_id = class_uri_to_node_id[class_uri]
                
                # Add subClassOf edges
                for superclass_uri in class_data.get('subclass_of', []):
                    if superclass_uri in class_uri_to_node_id:
                        target_node_id = class_uri_to_node_id[superclass_uri]
                        edge = {
                            "source": source_node_id,
                            "target": target_node_id,
                            "label": "rdfs:subClassOf",
                            "graph_type": "ontology_graph",
                            "relationship_type": "subClassOf",
                            "source_class_uri": class_uri,
                            "target_class_uri": superclass_uri
                        }
                        self.merged_graph["links"].append(edge)
                        ontological_edges_added += 1
                
                # Add equivalent class relationships
                for equiv_uri in class_data.get('equivalent_classes', []):
                    if equiv_uri in class_uri_to_node_id:
                        target_node_id = class_uri_to_node_id[equiv_uri]
                        edge = {
                            "source": source_node_id,
                            "target": target_node_id,
                            "label": "owl:equivalentClass",
                            "graph_type": "ontology_graph",
                            "relationship_type": "equivalentClass",
                            "source_class_uri": class_uri,
                            "target_class_uri": equiv_uri
                        }
                        self.merged_graph["links"].append(edge)
                        ontological_edges_added += 1
                
                # Add disjoint class relationships
                for disjoint_uri in class_data.get('disjoint_with', []):
                    if disjoint_uri in class_uri_to_node_id:
                        target_node_id = class_uri_to_node_id[disjoint_uri]
                        edge = {
                            "source": source_node_id,
                            "target": target_node_id,
                            "label": "owl:disjointWith",
                            "graph_type": "ontology_graph",
                            "relationship_type": "disjointWith",
                            "source_class_uri": class_uri,
                            "target_class_uri": disjoint_uri
                        }
                        self.merged_graph["links"].append(edge)
                        ontological_edges_added += 1
        
        # 2. Add object property relationships (domain -> property -> range)
        # Add object properties as nodes and connect to their domain/range classes
        object_properties = self.ontology.get('object_properties', {})
        property_nodes_added = 0
        
        for prop_uri, prop_data in object_properties.items():
            # Only include properties that connect our used classes
            domains = prop_data.get('domain', [])
            ranges = prop_data.get('range', [])
            
            # Check if this property connects our used classes
            has_used_domain = any(domain_uri in used_ontology_classes for domain_uri in domains)
            has_used_range = any(range_uri in used_ontology_classes for range_uri in ranges)
            
            if has_used_domain and has_used_range:
                # Add object property as a node
                prop_node_id = f"PROP_{prop_uri.split('#')[-1].split('/')[-1]}"
                prop_node = {
                    "node_type": "OBJECT_PROPERTY",
                    "graph_type": "ontology_graph", 
                    "uri": prop_uri,
                    "label": prop_data.get('label', ''),
                    "comment": prop_data.get('comment', ''),
                    "ontology_type": prop_data.get('ontology_type', 'unknown'),
                    "functional": prop_data.get('functional', False),
                    "transitive": prop_data.get('transitive', False),
                    "symmetric": prop_data.get('symmetric', False),
                    "id": prop_node_id
                }
                self.merged_graph["nodes"].append(prop_node)
                property_nodes_added += 1
                
                # Add domain edges (class -> property)
                for domain_uri in domains:
                    if domain_uri in class_uri_to_node_id:
                        domain_node_id = class_uri_to_node_id[domain_uri]
                        edge = {
                            "source": domain_node_id,
                            "target": prop_node_id,
                            "label": "rdfs:domain",
                            "graph_type": "ontology_graph",
                            "relationship_type": "domain",
                            "class_uri": domain_uri,
                            "property_uri": prop_uri
                        }
                        self.merged_graph["links"].append(edge)
                        ontological_edges_added += 1
                
                # Add range edges (property -> class)  
                for range_uri in ranges:
                    if range_uri in class_uri_to_node_id:
                        range_node_id = class_uri_to_node_id[range_uri]
                        edge = {
                            "source": prop_node_id,
                            "target": range_node_id,
                            "label": "rdfs:range",
                            "graph_type": "ontology_graph",
                            "relationship_type": "range",
                            "property_uri": prop_uri,
                            "class_uri": range_uri
                        }
                        self.merged_graph["links"].append(edge)
                        ontological_edges_added += 1
        
        # Update metadata
        if "metadata" not in self.merged_graph:
            self.merged_graph["metadata"] = {}
        
        self.merged_graph["metadata"].update({
            "merge_timestamp": datetime.now().isoformat(),
            "ontology_classes_added": len(used_ontology_classes),
            "object_properties_added": property_nodes_added,
            "example_of_edges_added": example_of_edges_added,
            "ontological_edges_added": ontological_edges_added,
            "total_mappings": len(mappings),
            "source_knowledge_graph": self.knowledge_graph_path,
            "source_ontology": self.ontology_path,
            "merger_version": "2.0"
        })
        
        logger.info(f"Merged graph created:")
        logger.info(f"  - Added {len(used_ontology_classes)} ontology class nodes")
        logger.info(f"  - Added {property_nodes_added} object property nodes")
        logger.info(f"  - Added {example_of_edges_added} EXAMPLE_OF edges") 
        logger.info(f"  - Added {ontological_edges_added} ontological relationship edges")
        logger.info(f"  - Total nodes: {len(self.merged_graph['nodes'])}")
        logger.info(f"  - Total edges: {len(self.merged_graph.get('links', []))}")
        
        return self.merged_graph
    
    def save_merged_graph(self, output_path: str) -> None:
        """Save the merged graph to JSON file"""
        logger.info(f"Saving merged graph to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.merged_graph, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Merged graph saved successfully")
        
    async def merge(self, output_path: str = "merged_knowledge_ontology.json") -> bool:
        """Main method to perform the merge"""
        logger.info("Starting knowledge graph to ontology merge process...")
        
        # Step 1: Load graphs
        if not self.load_graphs():
            return False
        
        # Step 2: Extract entities and concepts
        self.extract_entities_concepts()
        
        # Step 3: Extract ontology classes
        self.extract_ontology_classes()
        
        # Step 4: Create batches
        batches = self.create_intelligent_batches()
        
        if not batches:
            logger.warning("No batches created - no entities/concepts to process")
            return False
        
        # Step 5: Process with LLM
        mappings = await self.process_all_batches(batches)
        
        if not mappings:
            logger.warning("No mappings found - no relationships to add")
            # Continue anyway to save the graph structure
        
        # Step 6: Create merged graph
        self.create_merged_graph(mappings if mappings else [])
        
        # Step 7: Save result
        self.save_merged_graph(output_path)
        
        logger.info("Merge process completed successfully!")
        return True

async def main():
    """Main function for testing"""
    # File paths - all in output folder for clean organization
    knowledge_graph_path = "../output/graph_community.json"  # From kg.py
    ontology_path = "../output/eu_cdm_ontology.json"      # From rdf_to_ontology_json.py  
    output_path = "../output/merged_knowledge_ontology.json"  # Keep output organized
    
    # Check if files exist
    if not os.path.exists(knowledge_graph_path):
        logger.error(f"Knowledge graph file not found: {knowledge_graph_path}")
        logger.info("Run kg.py first to generate the knowledge graph")
        return
    
    if not os.path.exists(ontology_path):
        logger.error(f"Ontology file not found: {ontology_path}")
        logger.info("Run rdf_to_ontology_json.py first to generate the ontology")
        return
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set")
        return
    
    # Create merger and run
    merger = KnowledgeOntologyMerger(knowledge_graph_path, ontology_path)
    
    success = await merger.merge(output_path)
    
    if success:
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH TO ONTOLOGY MERGE COMPLETED!")
        print("="*60)
        print(f"Input Knowledge Graph: {knowledge_graph_path}")
        print(f"Input Ontology: {ontology_path}")
        print(f"Output Merged Graph: {output_path}")
        print("\nThe merged graph contains:")
        print("- All original knowledge graph nodes and edges")
        print("- Selected ontology class nodes")
        print("- New EXAMPLE_OF edges connecting entities to ontology classes")
        print("\nOpen the merged graph to explore entity-to-class relationships!")
    else:
        print("Merge process failed. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main()) 