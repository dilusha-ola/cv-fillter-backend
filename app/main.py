from fastapi import FastAPI
from app.api.v1.upload import router as upload_router
from app.api.v1.filter import router as filter_router
from app.core.security import setup_cors

app = FastAPI(
    title="CV Filtering System API",
    description="Backend API for uploading, vectorizing, and screening candidate CVs against target requirements using LangChain and OpenAI.",
    version="1.0.0"
)

# Setup CORS middleware
setup_cors(app)

# Register API routers with the global prefix
app.include_router(upload_router, prefix="/api/v1", tags=["Upload"])
app.include_router(filter_router, prefix="/api/v1", tags=["Filter"])

@app.get("/")
async def root():
    """
    Root endpoint for health checking the API service.
    """
    return {
        "status": "healthy",
        "message": "CV Filtering System API is up and running."
    }
