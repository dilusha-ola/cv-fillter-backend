from pydantic import BaseModel, Field
from typing import List, Literal

class ConditionCheck(BaseModel):
    condition: str = Field(..., description="The screening condition evaluated")
    status: Literal["PASSED", "FAILED"] = Field(..., description="Status check status: PASSED if met, FAILED if not met")
    evidence: str = Field(..., description="Direct quote or clear evidence found from the CV context")

class CVAnalysisResponse(BaseModel):
    candidate_name: str = Field(..., description="The full name of the candidate extracted from the CV")
    overall_match_score: int = Field(..., description="An overall match score out of 100 based on job requirements and conditions")
    condition_checks: List[ConditionCheck] = Field(..., description="Evaluation of the strict conditions")
    summary: str = Field(..., description="An AI-generated screening synthesis of strengths, weaknesses and gaps")
