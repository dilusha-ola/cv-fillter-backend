from fastapi import APIRouter, Header, HTTPException, status
from app.schemas.request import FilterRequest
from app.schemas.response import CVAnalysisResponse
from app.services import vector_store, chain
from app.core.config import settings

router = APIRouter()

@router.post("/analyze", response_model=CVAnalysisResponse)
async def analyze_cv(
    request: FilterRequest,
    authorization: str = Header(None)
):
    """
    Analyzes an indexed CV using hybrid context retrieval and LangChain LLM evaluation 
    against a target position, JD, and a list of strict screening criteria.
    """
    # 1. Determine LLM API Key (either from request header or .env)
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.split(" ")[1]
        
    if not api_key:
        if settings.LLM_PROVIDER.lower() in ["groq"]:
            api_key = settings.GROQ_API_KEY
        else:
            api_key = settings.OPENAI_API_KEY
            
    if not api_key or api_key in ["your_openai_api_key_here", "your_groq_api_key_here"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API key for {settings.LLM_PROVIDER} must be provided in the Authorization header or set in the backend environment variables."
        )
        
    try:
        # 2. Retrieve relevant chunks from the indexed CV
        chunks = vector_store.retrieve_combined_chunks(
            cv_id=request.cv_id,
            position=request.position,
            jd=request.jd,
            conditions=request.conditions,
            api_key=api_key
        )
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No CV text context found for CV ID: {request.cv_id} to evaluate."
            )
            
        # 3. Analyze CV context chunks against the JD and criteria using the LLM
        evaluation = chain.analyze_cv_with_llm(
            retrieved_chunks=chunks,
            position=request.position,
            jd=request.jd,
            conditions=request.conditions,
            api_key=api_key
        )
        
        return evaluation
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during CV analysis: {str(e)}"
        )
