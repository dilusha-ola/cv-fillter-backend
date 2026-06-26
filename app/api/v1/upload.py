from fastapi import APIRouter, UploadFile, File, Header, HTTPException, status
import uuid
import os
import shutil
from app.core.config import settings
from app.services import document, vector_store

router = APIRouter()

@router.post("/upload")
async def upload_cv(
    file: UploadFile = File(...), 
    authorization: str = Header(None)
):
    """
    Accepts a CV file (PDF or DOCX), extracts text, chunks it, and 
    generates embeddings to save into the vector store.
    """
    # 1. Determine OpenAI API Key (either from request header or .env)
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.split(" ")[1]
        
    if not api_key:
        api_key = settings.OPENAI_API_KEY
        
    if not api_key or api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI API key must be provided in the Authorization header or set in the backend environment variables."
        )
        
    # 2. Validate file format extension
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a PDF or Word (.docx) document."
        )
        
    # 3. Setup folders
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
    
    # 4. Generate unique ID for this CV instance
    cv_id = str(uuid.uuid4())
    temp_file_path = os.path.join(settings.UPLOAD_DIR, f"{cv_id}{ext}")
    
    try:
        # 5. Save raw file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 6. Extract text and segment into chunks
        chunks = document.load_and_chunk_document(temp_file_path)
        
        if not chunks:
            raise ValueError("No readable text content could be extracted from the uploaded document.")
            
        # 7. Embed chunks and save vector index
        vector_store.create_and_save_vector_store(
            chunks=chunks,
            cv_id=cv_id,
            openai_api_key=api_key
        )
        
        return {
            "cv_id": cv_id,
            "filename": filename,
            "message": "CV uploaded, processed, and vectorized successfully."
        }
        
    except Exception as e:
        # Clean up temporary uploaded file if error occurs
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and vectorize CV: {str(e)}"
        )
