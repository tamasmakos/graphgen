from typing import List, Dict, Any, Optional
import networkx as nx
from pydantic import BaseModel, Field

class ChunkExtractionTask(BaseModel):
    """
    Represents a task to extract entities and relations from a text chunk.
    """
    chunk_id: str
    chunk_text: str
    entities: List[str] = Field(default_factory=list)
    abstract_concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

class SegmentData(BaseModel):
    """
    Data model for a document segment.
    """
    segment_id: str
    content: str
    line_number: int
    date: Any # datetime.date but Any for simplicity in pydantic
    sentiment: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)



class PipelineContext:
    """
    The "Bus" / Context object that holds the state of the pipeline execution.
    It passes the graph and shared stats between steps.
    """
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph if graph is not None else nx.DiGraph()
        self.extraction_tasks: List[ChunkExtractionTask] = []
        self.stats: Dict[str, Any] = {}
        self.errors: Dict[str, List[str]] = {}
        self.total_segments: int = 0
        
    def add_error(self, stage: str, message: str):
        """Register a non-fatal error for reporting."""
        if stage not in self.errors:
            self.errors[stage] = []
        self.errors[stage].append(message)
