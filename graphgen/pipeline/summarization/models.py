from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SummarizationTask:
    task_id: str
    community_id: int
    subcommunity_id: Optional[int]
    is_topic: bool
    concatenated_text: str
    chunk_ids: List[str]
    entity_ids: List[str]
    title: Optional[str] = None
    summary: Optional[str] = None
