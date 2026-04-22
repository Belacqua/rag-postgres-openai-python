from enum import Enum
from typing import Any, Optional

from openai.types.responses import ResponseInputItemParam
from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    TEXT = "text"
    VECTORS = "vectors"
    HYBRID = "hybrid"


class ChatRequestOverrides(BaseModel):
    top: int = 3
    temperature: float = 0.3
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    use_advanced_flow: bool = True
    prompt_template: Optional[str] = None


class ChatRequestContext(BaseModel):
    overrides: ChatRequestOverrides


class ChatRequest(BaseModel):
    input: list[ResponseInputItemParam]
    context: ChatRequestContext


class CapabilityPublic(BaseModel):
    id: int
    subcategory_name: str
    subcategory_description: str
    classification_name: str
    category_name: str
    naics_code: Optional[str] = None

    def to_str_for_rag(self):
        return (
            f"Classification:{self.classification_name} "
            f"Category:{self.category_name} "
            f"Subcategory:{self.subcategory_name} "
            f"Description:{self.subcategory_description} "
            f"NAICS:{self.naics_code}"
        )


class CapabilityWithDistance(CapabilityPublic):
    distance: float

    def __init__(self, **data):
        super().__init__(**data)
        self.distance = round(self.distance, 2)


class ThoughtStep(BaseModel):
    title: str
    description: Any
    props: dict = {}


class RAGContext(BaseModel):
    data_points: dict[int, CapabilityPublic]
    thoughts: list[ThoughtStep]


class ErrorResponse(BaseModel):
    error: str


class RetrievalResponse(BaseModel):
    output_text: str
    context: RAGContext


class RetrievalResponseDelta(BaseModel):
    type: str
    delta: Optional[str] = None
    context: Optional[RAGContext] = None


class ChatParams(ChatRequestOverrides):
    prompt_template: str
    response_token_limit: int = 1024
    enable_text_search: bool
    enable_vector_search: bool
    original_user_query: str
    past_messages: list[ResponseInputItemParam]


class Filter(BaseModel):
    column: str
    comparison_operator: str
    value: Any


class ClassificationFilter(Filter):
    column: str = Field(default="classification_name", description="Always 'classification_name'")
    comparison_operator: str = Field(description="Operator for comparison ('=' or '!=')")
    value: str = Field(description="Classification name, e.g. 'IT Services' or 'Professional Services'")


class CategoryFilter(Filter):
    column: str = Field(default="category_name", description="Always 'category_name'")
    comparison_operator: str = Field(description="Operator for comparison ('=' or '!=')")
    value: str = Field(description="Category name, e.g. 'Engineering' or 'Analytics'")


class SearchResults(BaseModel):
    query: str
    items: list[CapabilityPublic]
    filters: list[Filter]
