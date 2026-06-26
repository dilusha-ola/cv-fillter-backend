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
        "You are a strict HR screening AI. You follow rules exactly. You do NOT infer, assume, or guess.\n\n"

        "=== STEP 1: EXTRACT CANDIDATE NAME ===\n"
        "Find the candidate's full name in the CV text. If absent, output 'Unknown Candidate'.\n\n"

        "=== STEP 2: EVALUATE EACH CONDITION USING THIS EXACT DECISION TREE ===\n"
        "For EVERY condition, walk through these questions in order:\n\n"
        "Q1. Is the required skill, technology, or subject mentioned ANYWHERE in the CV?\n"
        "    → NO  → status = FAILED. Stop.\n"
        "    → YES → continue to Q2.\n\n"
        "Q2. Does the condition contain a NUMBER (years, months, count, level, etc.)?\n"
        "    → NO  → The skill is present and no quantity is required → status = PASSED. Stop.\n"
        "    → YES → continue to Q3.\n\n"
        "Q3. Does the CV contain an EXPLICIT numeric statement that DIRECTLY satisfies the number?\n"
        "    Accepted evidence examples:\n"
        "      - '3 years of React experience'\n"
        "      - 'React developer since January 2022' (if calculable to meet the requirement)\n"
        "      - 'Jan 2021 – Dec 2023 | React Developer' (date range = 3 years)\n"
        "    NOT accepted (these must NOT result in PASSED):\n"
        "      - Listing React in a skills section with no dates\n"
        "      - A project that used React with no employment dates\n"
        "      - Using Next.js, Remix, or any React-based framework without stating React years\n"
        "      - Any sentence that does not contain a number or date range\n"
        "    → EXPLICIT NUMERIC PROOF FOUND → status = PASSED. Stop.\n"
        "    → NO EXPLICIT NUMERIC PROOF  → status = NEUTRAL. Stop.\n\n"

        "=== STEP 3: SCORE (0–100) ===\n"
        "Compute two sub-scores and add them:\n"
        "  A) JD Alignment (max 40 pts): How well does the candidate's background match the\n"
        "     job description and target position overall? Be honest and critical.\n"
        "  B) Conditions (max 60 pts): Divide 60 pts equally across all conditions.\n"
        "     Each condition is worth (60 / total_conditions) pts.\n"
        "     PASSED = full pts, NEUTRAL = half pts, FAILED = 0 pts.\n"
        "  Final score = round(A + B).\n\n"

        "=== STEP 4: EVIDENCE ===\n"
        "For each condition provide a SHORT evidence string:\n"
        "  - PASSED:  Quote the exact text that proves it.\n"
        "  - NEUTRAL: Quote what was found, then state what numeric proof is missing.\n"
        "  - FAILED:  State that the skill/qualification was not found in the CV.\n\n"

        "=== STEP 5: SUMMARY ===\n"
        "Write a concise professional summary covering JD fit, key strengths,\n"
        "NEUTRAL items that need verification, and any hard disqualifiers."
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
