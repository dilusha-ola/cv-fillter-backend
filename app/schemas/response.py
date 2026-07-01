from pydantic import BaseModel, Field
from typing import List, Literal


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
    confidence: int = Field(
        ...,
        description=(
            "Confidence level 0-100 in the assigned status. "
            "90-100: unambiguous proof or absence; "
            "70-89: strong evidence with minor inference or semantic equivalence used; "
            "50-69: moderate evidence, some interpretation required; "
            "30-49: weak or indirect evidence; "
            "0-29: highly uncertain."
        ),
        ge=0,
        le=100,
    )
    evidence: str = Field(
        ...,
        description="Exact CV text used for evaluation, or a clear statement of absence if FAILED",
    )
    reasoning: str = Field(
        ...,
        description="1-2 sentences explaining how the evidence was interpreted to reach this status and confidence level",
    )


class JDAlignmentBreakdown(BaseModel):
    technical: int = Field(..., description="Technical skills/tools/stack alignment score 0-8", ge=0, le=8)
    industry: int = Field(..., description="Industry/sector background match score 0-4", ge=0, le=4)
    experience: int = Field(..., description="Seniority and scope-of-experience fit score 0-4", ge=0, le=4)
    education: int = Field(..., description="Degree, certifications, and formal training alignment score 0-2", ge=0, le=2)
    domain: int = Field(..., description="Specific domain/vertical knowledge match score 0-2", ge=0, le=2)
    total: int = Field(..., description="Sum of all five sub-scores, 0-20", ge=0, le=20)
    reasons: str = Field(
        ...,
        description="2-4 sentence explanation covering key strengths and gaps across the five alignment dimensions",
    )


class CVAnalysisResponse(BaseModel):
    candidate_name: str = Field(..., description="The full name of the candidate extracted from the CV")
    overall_match_score: int = Field(
        ...,
        description="Overall match score 0-100: jd_alignment.total (0-20) plus conditions score (0-80)",
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
