from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas.response import CVAnalysisResponse
from app.core.config import settings
from typing import List
from langchain_core.documents import Document

def analyze_cv_with_llm(
    retrieved_chunks: List[Document],
    position: str,
    jd: str,
    conditions: List[str],
    api_key: str
) -> CVAnalysisResponse:
    """
    Combines retrieved CV chunks, prompt inputs, and strict criteria to produce
    a type-safe, validated screening report from the selected LLM provider.
    """
    # 1. Format retrieved CV context chunks
    context_text = "\n\n---\n\n".join([
        f"[Chunk {idx+1}]: {doc.page_content}" 
        for idx, doc in enumerate(retrieved_chunks)
    ])
    
    # 2. Set up prompts
    system_prompt = (
        "You are a strict, evidence-based HR screening AI. You evaluate CVs with high precision.\n\n"
        "=== CONDITION EVALUATION RULES ===\n"
        "For each condition, you MUST assign one of: PASSED, FAILED, or NEUTRAL.\n\n"
        "PASSED — Only when the CV contains EXPLICIT, DIRECT, WORD-FOR-WORD proof:\n"
        "  • For duration/numeric requirements (e.g. '3+ years of React', '5+ years experience'):\n"
        "    The CV must contain an explicit statement of years/months (e.g. '3 years of React',\n"
        "    'React developer since 2021', or date ranges that calculate to the required duration).\n"
        "    Using a framework (Next.js, Remix) that is BUILT ON React does NOT prove React years.\n"
        "    Project names, tech stacks listed without dates, or implied usage do NOT count as PASSED.\n"
        "  • For skill requirements (e.g. 'Must know TypeScript'):\n"
        "    The skill must be explicitly named. Adjacent or related skills are NOT sufficient.\n\n"
        "NEUTRAL — When the skill/technology IS mentioned but the required quantity/level is unverifiable:\n"
        "  • Candidate lists React as a skill or uses it in a project, but states no years.\n"
        "  • Candidate has relevant experience but dates are missing or ambiguous.\n"
        "  • You can confirm awareness but NOT the required depth or duration.\n\n"
        "FAILED — When the required skill, technology, or qualification is completely absent from the CV.\n\n"
        "=== SCORING RULES ===\n"
        "Compute overall_match_score (0–100) using TWO components:\n"
        "  Component A — JD Alignment (40% of score):\n"
        "    How well does the candidate's overall background, seniority, and experience match\n"
        "    the Job Description and target position? Score 0–40.\n"
        "  Component B — Conditions Score (60% of score):\n"
        "    PASSED condition = full points, NEUTRAL = 50% points, FAILED = 0 points.\n"
        "    Score = (sum of condition points / max points) * 60.\n"
        "  Final score = Component A + Component B (rounded to nearest integer).\n\n"
        "=== OTHER RULES ===\n"
        "1. Extract candidate full name. If not found, output 'Unknown Candidate'.\n"
        "2. Evidence must be a direct quote or a precise factual statement from the CV text.\n"
        "   For NEUTRAL/FAILED, explain concisely what is missing.\n"
        "3. Summary must cover: overall JD fit, key strengths, NEUTRAL gaps that need verification,\n"
        "   and clear disqualifiers."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", (
            "Target Job Position: {position}\n\n"
            "Job Description (JD):\n{jd}\n\n"
            "Strict Conditions to Check:\n{conditions_list}\n\n"
            "CV Context Snippets:\n{context}\n\n"
            "Please perform the evaluation and return the structured response."
        ))
    ])
    
    # Format conditions as a list
    conditions_list_str = "\n".join([f"- {c}" for c in conditions])
    
    # 3. Instantiate LLM based on configured provider
    provider = settings.LLM_PROVIDER.lower()
    if provider == "groq":
        llm = ChatOpenAI(
            model=settings.GROQ_MODEL,
            temperature=0.0,
            openai_api_key=api_key or settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            openai_api_key=api_key or settings.OPENAI_API_KEY
        )
    
    # 4. Bind strict Pydantic model for output parsing (use function_calling for Groq compatibility)
    structured_llm = llm.with_structured_output(CVAnalysisResponse, method="function_calling")
    
    # 5. Create chain & execute
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "position": position,
        "jd": jd,
        "conditions_list": conditions_list_str,
        "context": context_text
    })
    
    return result
