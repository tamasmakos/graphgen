from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class NodeSchema(BaseModel):
    label: str
    attributes: List[str] = Field(default_factory=list)
    # Mapping rule: how to derive this node from a Segment or Doc
    # simple mapping for now: e.g. "segment" means 1 node per segment
    source_type: str = "segment" # "segment", "document", "chunk", "derived"

class EdgeSchema(BaseModel):
    source_label: str
    target_label: str
    relation_type: str
    # If deriving from hierarchy, e.g. Doc -> Chunk
    is_hierarchical: bool = False 

class GraphSchema(BaseModel):
    """
    Defines the target structure of the knowledge graph.
    """
    nodes: Dict[str, NodeSchema] # Keyed by internal name (e.g. "Doc", "Chunk")
    edges: List[EdgeSchema] = Field(default_factory=list)

