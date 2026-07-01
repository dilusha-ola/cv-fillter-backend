from pydantic import BaseModel, Field
from typing import List

class FilterRequest(BaseModel):
    cv_id: str = Field(..., description="Unique UUID of the uploaded CV vector store")
    position: str = Field(..., description="Target Job Position")
    jd: str = Field(..., description="Full Job Description details")
    conditions: List[str] = Field(..., description="Strict criteria to evaluate candidate against")
