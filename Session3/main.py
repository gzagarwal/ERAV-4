from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer
import os
import shutil
from pathlib import Path
from typing import Optional, List
import uvicorn
from dotenv import load_dotenv

from models import (
    QuestionRequest,
    CodeGenerationRequest,
    ParagraphAnalysisRequest,
    TechnicalAnalysisRequest,
    UploadResponse,
    QuestionResponse,
    CodeResponse,
    AnalysisResponse,
    TechnicalAnalysisResponse,
)
from services.pdf_service import PDFService
from services.ai_service import AIService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper Analyzer",
    description="A powerful AI-powered research paper analysis tool using Gemini API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize services
pdf_service = PDFService()
ai_service = AIService()

# Mount static files for React build
if Path("client/build").exists():
    app.mount("/static", StaticFiles(directory="client/build/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    """Serve the React frontend"""
    if Path("client/build/index.html").exists():
        with open("client/build/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Research Paper Analyzer API</h1><p>Frontend not built yet. Visit <a href='/docs'>/docs</a> for API documentation.</p>"
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "message": "Research Paper Analyzer API is running"}


@app.post("/api/upload-pdf", response_model=UploadResponse)
async def upload_and_analyze_pdf(
    pdf: UploadFile = File(..., description="PDF file to analyze"),
):
    """
    Upload and analyze a research paper PDF

    - **pdf**: PDF file containing the research paper
    - Returns: Extracted text and AI-generated summary
    """
    try:
        # Validate file type
        if not pdf.content_type == "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed",
            )

        # Validate file size (10MB limit)
        if pdf.size and pdf.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size too large. Maximum 10MB allowed.",
            )

        # Save uploaded file
        file_path = UPLOAD_DIR / f"{pdf.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(pdf.file, buffer)

        # Extract text from PDF
        text = await pdf_service.extract_text(str(file_path))

        # Generate AI summary
        summary = await ai_service.generate_summary(text)

        # Clean up uploaded file
        os.remove(file_path)

        return UploadResponse(
            success=True,
            filename=pdf.filename,
            text=text,
            summary=summary,
            message="PDF uploaded and analyzed successfully",
        )

    except Exception as e:
        # Clean up file if it exists
        if "file_path" in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF: {str(e)}",
        )


@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a specific question about the research paper

    - **question**: Your question about the paper
    - **paper_text**: The full text of the research paper
    - Returns: AI-generated answer to your question
    """
    try:
        response = await ai_service.answer_question(
            request.question, request.paper_text
        )
        return QuestionResponse(success=True, response=response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}",
        )


@app.post("/api/generate-code", response_model=CodeResponse)
async def generate_code(request: CodeGenerationRequest):
    """
    Generate code implementation based on the research paper

    - **algorithm**: Description of the algorithm to implement
    - **language**: Programming language (python, java, go, etc.)
    - **paper_text**: The full text of the research paper
    - Returns: Generated code with explanations
    """
    try:
        code = await ai_service.generate_code(
            request.algorithm, request.language, request.paper_text
        )
        return CodeResponse(success=True, code=code)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating code: {str(e)}",
        )


@app.post("/api/analyze-paragraph", response_model=AnalysisResponse)
async def analyze_paragraph(request: ParagraphAnalysisRequest):
    """
    Analyze a specific paragraph from the research paper

    - **paragraph**: The paragraph text to analyze
    - **paper_text**: The full text of the research paper
    - Returns: Detailed analysis of the paragraph
    """
    try:
        analysis = await ai_service.analyze_paragraph(
            request.paragraph, request.paper_text
        )
        return AnalysisResponse(success=True, analysis=analysis)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing paragraph: {str(e)}",
        )


@app.post("/api/technical-analysis", response_model=TechnicalAnalysisResponse)
async def get_technical_analysis(request: TechnicalAnalysisRequest):
    """
    Get detailed technical analysis of the research paper

    - **paper_text**: The full text of the research paper
    - Returns: Comprehensive technical analysis
    """
    try:
        technical_analysis = await ai_service.get_technical_analysis(request.paper_text)
        return TechnicalAnalysisResponse(
            success=True, technical_analysis=technical_analysis
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating technical analysis: {str(e)}",
        )


# Catch-all route for React Router
@app.get("/{full_path:path}")
async def serve_react_app_catchall(full_path: str):
    """Serve React app for all other routes (for client-side routing)"""
    if Path("client/build/index.html").exists():
        with open("client/build/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Frontend not available"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )
