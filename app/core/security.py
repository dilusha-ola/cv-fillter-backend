from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def setup_cors(app: FastAPI):
    # Configure CORS so that Next.js frontend can make API calls to FastAPI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust this in production to match your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
