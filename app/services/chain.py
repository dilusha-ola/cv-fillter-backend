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
    api_key: str,
    web_profile_data: str = "",
) -> CVAnalysisResponse:
    """
    Combines retrieved CV chunks, prompt inputs, and strict criteria to produce
    a type-safe, validated screening report from the selected LLM provider.
    Optional web_profile_data (GitHub repos, portfolio text) is appended as
    supplementary evidence alongside the CV context.
    """
    # 1. Format retrieved CV context chunks
    context_text = "\n\n---\n\n".join([
        f"[Chunk {idx+1}]: {doc.page_content}" 
        for idx, doc in enumerate(retrieved_chunks)
    ])
    
    # 2. Set up prompts
    system_prompt = (
        "You are a professional HR screening AI. You are precise, honest, and thorough.\n"
        "You accept semantic equivalents but never make unsupported assumptions.\n"
        "IMPORTANT: All numeric fields (confidence, technical, industry, experience, education, domain, total) "
        "MUST be plain integers \u2014 write 4 not \"4\", write 80 not \"80\".\n"
        "If 'Additional Web Profile Data' is provided (GitHub repos, portfolio content), treat it as "
        "supplementary CV evidence with equal weight to the uploaded CV text.\n\n"

        "=== STEP 1: EXTRACT CANDIDATE NAME ===\n"
        "Find the candidate's full name in the CV text. If absent, output 'Unknown Candidate'.\n\n"

        "=== STEP 2: EVALUATE EACH CONDITION ===\n"
        "For EVERY condition, follow this decision tree STRICTLY in order:\n\n"
        "Q1. SKILL / TECHNOLOGY PRESENCE CHECK (completely ignore any quantity or duration here):\n"
        "    Is the underlying skill, technology, or qualification present ANYWHERE in the CV?\n"
        "    Accept: exact name, a recognized synonym, or a functionally equivalent skill.\n"
        "    Examples — ALL of these pass Q1 regardless of duration:\n"
        "      'AWS Cloud' condition   ← 'AWS' listed in skills section ✓\n"
        "      'AWS Cloud' condition   ← 'AWS Cloud Practitioner certification' ✓\n"
        "      'REST API experience'   ← 'Spring Boot services exposing HTTP endpoints' ✓\n"
        "      'SQL experience'        ← 'PostgreSQL and MySQL listed in projects' ✓\n"
        "      'team lead experience'  ← 'Led a team of 5 engineers' ✓\n"
        "    Reject only if the skill/technology is genuinely absent with no equivalent at all.\n"
        "    *** SPECIAL RULE — FORMAL QUALIFICATIONS (degrees, diplomas, certifications): ***\n"
        "    If the condition says 'hold', 'have', 'completed', or implies possession of a formal\n"
        "    qualification, the candidate must have COMPLETED and been AWARDED that qualification.\n"
        "    The following are UNAMBIGUOUS signals that a degree is NOT yet completed — assign FAILED immediately:\n"
        "      • Education entry ends with '– Present', '- Present', or 'Present' (no graduation date)\n"
        "      • CV text uses the word 'undergraduate' (e.g. 'I am a third-year undergraduate')\n"
        "      • CV text uses 'currently pursuing', 'currently studying', 'currently enrolled'\n"
        "      • Education entry shows 'Expected graduation YYYY' or 'Expected YYYY'\n"
        "      • A start year is given but no end year (e.g. '2023 –' with nothing after)\n"
        "    Do NOT assign NEUTRAL when these signals are present — they are definitive proof the\n"
        "    degree is incomplete. Assign FAILED with high confidence (80+).\n"
        "    Completed degree examples (Q1 passes ✓):\n"
        "      'B.Sc. Computer Science, 2019 – 2023' (past end year, no 'Present') ✓\n"
        "      'Bachelor's degree awarded June 2022' ✓\n"
        "    → Qualification/skill not found, or required degree is incomplete → status = FAILED. Stop.\n"
        "    → Skill found (or completed qualification confirmed) → continue to Q2.\n\n"
        "Q2. Does the condition contain an EXPLICIT REQUIRED QUANTITY?\n"
        "    A quantity is ONLY present when the condition itself states specific numbers:\n"
        "      '2+ years', 'at least 3 years', 'minimum N projects', 'N months', etc.\n"
        "    *** COMMON TRAPS — these are NOT quantity requirements → answer NO: ***\n"
        "      - 'React experience'                       → NO (skill just needs to exist)\n"
        "      - 'experience with Docker'                 → NO (skill just needs to exist)\n"
        "      - 'Bachelor's degree in CS or similar'     → NO (qualification just needs to exist;\n"
        "                                                       'or similar' means equivalents qualify)\n"
        "      - 'AWS knowledge'                          → NO (skill just needs to exist)\n"
        "      - 'TypeScript skills'                      → NO (skill just needs to exist)\n"
        "    Only answer YES for conditions like:\n"
        "      - 'Must have 2+ years of React experience' → YES\n"
        "      - 'At least 3 projects using AWS'          → YES\n"
        "      - 'Minimum 5 years backend development'    → YES\n"
        "    → NO  → Skill/qualification is present and no quantity was required → status = PASSED. Stop.\n"
        "    → YES → continue to Q3.\n\n"
        "Q3. Does the CV contain EXPLICIT quantitative proof that meets or exceeds the required quantity?\n"
        "    Accepted proof:\n"
        "      - '3 years of React experience'\n"
        "      - 'React Developer since January 2022' (date is calculable and meets requirement)\n"
        "      - 'Jan 2021 – Dec 2023 | React Developer' (date range ≥ required years)\n"
        "    NOT accepted as proof:\n"
        "      - Skill listed in a skills section with no employment dates or duration\n"
        "      - A certification in the skill with no stated work duration\n"
        "      - A project using the skill with no role start/end dates\n"
        "    *** CRITICAL RULE: If the skill was found in Q1 but no explicit quantity exists → status = NEUTRAL ***\n"
        "    This means: 'AWS' in skills section + condition requires '3 years AWS' + no duration stated = NEUTRAL.\n"
        "    → Explicit quantitative proof found → status = PASSED. Stop.\n"
        "    → Skill present but no explicit quantity for a quantity-required condition → status = NEUTRAL. Stop.\n\n"
        "For each condition also output:\n"
        "  evidence:   Quote the exact CV text found (skills entry, job title, date range, certification).\n"
        "              If FAILED, state exactly what was searched for and confirm it is absent.\n"
        "  reasoning:  3-4 sentences covering: (1) what was found in the CV, (2) why that evidence\n"
        "              does or does not satisfy the condition, (3) what specific proof is missing\n"
        "              if NEUTRAL, and (4) why this status was chosen over the alternatives.\n"
        "  confidence: Integer 0-100 for certainty in the assigned status.\n"
        "    90-100: Unambiguous explicit proof or unambiguous absence — no room for doubt.\n"
        "    70-89:  Strong evidence; minor inference or semantic equivalence used.\n"
        "    50-69:  Moderate evidence; some interpretation required.\n"
        "    30-49:  Weak or indirect evidence; significant inference needed.\n"
        "    0-29:   Highly uncertain; almost no relevant evidence found.\n\n"

        "=== STEP 3: JD ALIGNMENT BREAKDOWN (max 20 pts) ===\n"
        "Score the candidate across these five dimensions:\n"
        "  Technical Match  (0–8): How closely do technical skills, tools, and stack align with the JD?\n"
        "  Industry Match   (0–4): Does their industry/sector background match the JD context?\n"
        "  Experience Match (0–4): Does their seniority and scope of experience fit the role level?\n"
        "  Education        (0–2): Do their degree, certifications, and formal training align?\n"
        "  Domain Match     (0–2): Do they have the specific domain/vertical knowledge the JD targets?\n"
        "  total = technical + industry + experience + education + domain (0–20).\n"
        "  reasons: 4-6 sentences. For each dimension that scored below maximum, explain specifically\n"
        "    what was present in the CV and what was missing. Name actual skills, technologies,\n"
        "    certifications, or experience items. Do not write generic summaries.\n\n"

        "=== STEP 4: MISSING SKILLS ===\n"
        "Look ONLY at the STRICT CONDITIONS LIST above (not the JD).\n"
        "List the specific skills, tools, or qualifications from those conditions that are\n"
        "COMPLETELY ABSENT from the CV — i.e. conditions that resulted in FAILED status.\n"
        "Do NOT include skills that only appear in the JD text.\n"
        "Return short labels only, e.g. ['Kubernetes', 'Terraform'].\n"
        "If all conditions found at least some evidence, return an empty list.\n\n"

        "=== STEP 5: SUMMARY ==="
        "Write a concise professional screening summary covering JD fit, key strengths,\n"
        "NEUTRAL items that need further verification, and any hard disqualifiers."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", (
            "Target Job Position: {position}\n\n"
            "Job Description (JD):\n{jd}\n\n"
            "Strict Conditions to Check:\n{conditions_list}\n\n"
            "CV Context Snippets (primary source):\n{context}\n\n"
            "{web_profile_section}"
            "Please perform the evaluation and return the structured response."
        ))
    ])
    
    # Format conditions as a list
    conditions_list_str = "\n".join([f"- {c}" for c in conditions])

    # Build optional web profile section
    web_profile_section = ""
    if web_profile_data and web_profile_data.strip():
        web_profile_section = (
            f"Additional Web Profile Data (GitHub / Portfolio):\n{web_profile_data}\n\n"
        )
    
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
        "context": context_text,
        "web_profile_section": web_profile_section,
    })

    # 6. Compute overall_match_score in Python — LLMs are unreliable at arithmetic.
    #    Formula: JD alignment (0-20) + conditions score (0-80)
    #    Each condition worth (80 / N) pts: PASSED=full, NEUTRAL=half, FAILED=0
    n = len(result.condition_checks)
    if n > 0:
        pts_per = 80.0 / n
        conditions_score = sum(
            pts_per if c.status == "PASSED"
            else pts_per / 2.0 if c.status == "NEUTRAL"
            else 0.0
            for c in result.condition_checks
        )
    else:
        conditions_score = 0.0
    computed_score = round(result.jd_alignment.total + conditions_score)
    result = result.model_copy(update={"overall_match_score": computed_score})

    return result
