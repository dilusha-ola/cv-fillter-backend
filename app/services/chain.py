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
        "2. For each strict evaluation condition, decide if it is PASSED or FAILED:\n"
        "   - Set to 'PASSED' only if there is explicit or highly verifiable evidence in the CV context.\n"
        "   - Set to 'FAILED' if the CV lacks clear mention, lacks experience, or fails to prove the requirement.\n"
        "3. Provide brief, direct evidence or justification quotes from the CV for each check.\n"
        "4. Calculate an overall match score from 0 to 100 based on their experience and criteria overlap.\n"
        "5. Write a clean, professional synthesis summary detailing fit, key strengths, and missing requirements."
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
    if provider in ["grok", "xai"]:
        llm = ChatOpenAI(
            model=settings.GROK_MODEL,
            temperature=0.0,
            openai_api_key=api_key or settings.GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
    else:
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            openai_api_key=api_key or settings.OPENAI_API_KEY
        )
    
    # 4. Bind strict Pydantic model for output parsing
    structured_llm = llm.with_structured_output(CVAnalysisResponse)
    
    # 5. Create chain & execute
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "position": position,
        "jd": jd,
        "conditions_list": conditions_list_str,
        "context": context_text
    })
    
    return result
