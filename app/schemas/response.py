from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Literal, Union


def _to_int(v: Any) -> int:
    """Coerce string-wrapped or float integers from LLMs to plain int."""
    if isinstance(v, str):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0
    if isinstance(v, float):
        return int(v)
    return int(v) if v is not None else 0


class ConditionCheck(BaseModel):
    condition: str = Field(..., description="The screening condition evaluated")
    status: Literal["PASSED", "FAILED", "NEUTRAL"] = Field(
        ...,
        description=(
            "PASSED if credible evidence (direct or semantically equivalent) confirms the condition is met, "
            "including quantitative proof for numeric requirements; "
            "FAILED if no credible evidence in the CV satisfies the intent of the condition; "
            "NEUTRAL if relevant evidence exists but explicit quantitative proof is absent for a numeric requirement."
        ),
    )
    confidence: Union[int, str] = Field(
        ...,
        description=(
            "Confidence level 0-100 in the assigned status. "
            "Output as a plain integer, e.g. 80 not \"80\". "
            "90-100: unambiguous proof or absence; "
            "70-89: strong evidence with minor inference; "
            "50-69: moderate evidence; "
            "30-49: weak or indirect evidence; "
            "0-29: highly uncertain."
        ),
    )
    evidence: str = Field(
        ...,
        description="Exact CV text used for evaluation, or a clear statement of absence if FAILED",
    )
    reasoning: str = Field(
        ...,
        description="1-2 sentences explaining how the evidence was interpreted to reach this status and confidence level",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> int:
        return _to_int(v)


class JDAlignmentBreakdown(BaseModel):
    technical: Union[int, str] = Field(..., description="Technical skills/tools/stack alignment score 0-8. Plain integer.")
    industry: Union[int, str] = Field(..., description="Industry/sector background match score 0-4. Plain integer.")
    experience: Union[int, str] = Field(..., description="Seniority and scope-of-experience fit score 0-4. Plain integer.")
    education: Union[int, str] = Field(..., description="Degree, certifications, and formal training alignment score 0-2. Plain integer.")
    domain: Union[int, str] = Field(..., description="Specific domain/vertical knowledge match score 0-2. Plain integer.")
    total: Union[int, str] = Field(..., description="Sum of all five sub-scores, 0-20. Plain integer.")
    reasons: str = Field(
        ...,
        description="2-4 sentence explanation covering key strengths and gaps across the five alignment dimensions",
    )

    @field_validator("technical", "industry", "experience", "education", "domain", "total", mode="before")
    @classmethod
    def coerce_scores(cls, v: Any) -> int:
        return _to_int(v)


class CVAnalysisResponse(BaseModel):
    candidate_name: str = Field(..., description="The full name of the candidate extracted from the CV")
    overall_match_score: int = Field(
        0,
        description="Computed server-side: jd_alignment.total (0-20) + conditions score (0-80). Do not fill this in.",
        ge=0,
        le=100,
    )
    jd_alignment: JDAlignmentBreakdown = Field(
        ..., description="JD alignment score broken down across five dimensions"
    )
    condition_checks: List[ConditionCheck] = Field(..., description="Per-condition evaluation results")
    missing_skills: List[str] = Field(
        ...,
        description="Skills, tools, or qualifications from the JD/conditions completely absent from the CV",
    )
    summary: str = Field(
        ...,
        description="Concise professional screening synthesis covering JD fit, key strengths, NEUTRAL items needing verification, and hard disqualifiers",
    )
