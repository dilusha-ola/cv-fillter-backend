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
        "You are an expert HR recruitment specialist and CV screening AI.\n"
        "Your task is to analyze the candidate's CV snippets and evaluate their fit for the target job role.\n\n"
        "Evaluation Guidelines:\n"
        "1. Extract the candidate's full name from the context. If you cannot find a name, output 'Unknown Candidate'.\n"
        "2. For each strict evaluation condition, assign one of three statuses:\n"
        "   - PASSED: The CV contains explicit, verifiable proof the condition is fully met.\n"
        "             For numeric/duration requirements (e.g. '3+ years of React'), the CV must state\n"
        "             a clear timeframe or date range that satisfies the number.\n"
        "             Vague phrases like 'experienced in React' or listing React as a skill are NOT sufficient.\n"
        "   - FAILED: The CV does not mention the required skill, technology, or qualification at all.\n"
        "   - NEUTRAL: The candidate mentions the skill or technology but provides no explicit duration,\n"
        "              quantity, or measurable proof. Use this when you cannot confirm or deny a\n"
        "              numeric/level requirement (e.g. candidate lists React as a skill with no years stated).\n"
        "3. Provide brief, direct evidence or a justification quote from the CV text for each check.\n"
        "4. Calculate an overall match score from 0 to 100. Treat NEUTRAL conditions as partial credit (50%).\n"
        "5. Write a clean, professional synthesis summary detailing fit, key strengths, and gaps."
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
