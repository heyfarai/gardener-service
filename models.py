from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class TopicStatus(str, Enum):
    SEEDLING = "seedling"
    SAPLING = "sapling"
    MATURE = "mature"
    ARCHIVED = "archived"

class TopicMetadata(BaseModel):
    id: str
    label: str
    status: TopicStatus = Field(default=TopicStatus.SEEDLING)
    label_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    alias: List[str] = Field(default_factory=list)
    blurb: Optional[str] = None
    snippet_count: int = 0
    created_at: datetime
    updated_at: datetime

class Turn(BaseModel):
    model_config = {"protected_namespaces": ()}
    chat_id: str
    turn_id: str
    user_text: Optional[str] = ""
    model_text: Optional[str] = ""
    model: Optional[str] = None
    ts: Optional[str] = Field(default=None, description="Message timestamp")
    seedling: Optional[str] = None

class RetrieveReq(BaseModel):
    query: Optional[str] = None
    topic_id: Optional[str] = None
    k: int = 8
