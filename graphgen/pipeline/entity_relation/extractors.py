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

from graphgen.config.llm import get_langchain_llm

logger = logging.getLogger(__name__)



DEFAULT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at extracting knowledge graph entities and relationships from text.    
    Text:
    {input}
    """
)

class BaseExtractor(ABC):
    """Base class for graph extractors."""
    
    @abstractmethod
    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> Tuple[List[Tuple[str, str, str]], List[Dict[str, Any]]]:
        """
        Extract relations from text.
        
        Args:
            text: Text to extract relations from
            custom_prompt: Optional custom prompt template
            keywords: Optional list of keywords to guide extraction
            entities: Optional list of entities to focus on (used by LangChain)
            abstract_concepts: Optional list of abstract concepts (used by LangChain)
            
        Returns:
        Returns:
            Tuple containing:
            - List of (source, relation_type, target) triplets
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
    ) -> Tuple[List[Tuple[str, str, str]], List[Dict[str, Any]]]:
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
                        source = relationship.source.id
                        target = relationship.target.id
                        relation_type = relationship.type
                        relations.append((source, relation_type, target))
                        
                    # Extract nodes
                    for node in graph_doc.nodes:
                        nodes_data.append({
                            "id": node.id,
                            "type": node.type,
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
        
    extractor_type = extraction_config.get('backend', 'langchain') # fallback loop for now, we only have one
    
    logger.info(f"Initializing graph extractor: {extractor_type}")
    
    return LangChainExtractor(config)
