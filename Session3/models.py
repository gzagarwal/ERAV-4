from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """Request model for asking questions about the paper"""

    question: str = Field(
        ..., description="The question to ask about the research paper"
    )
    paper_text: str = Field(..., description="The full text of the research paper")


class CodeGenerationRequest(BaseModel):
    """Request model for code generation"""

    algorithm: str = Field(..., description="Description of the algorithm to implement")
    language: str = Field(
        ..., description="Programming language (python, java, go, etc.)"
    )
    paper_text: str = Field(..., description="The full text of the research paper")


class ParagraphAnalysisRequest(BaseModel):
    """Request model for paragraph analysis"""

    paragraph: str = Field(..., description="The paragraph text to analyze")
    paper_text: str = Field(..., description="The full text of the research paper")


class TechnicalAnalysisRequest(BaseModel):
    """Request model for technical analysis"""

    paper_text: str = Field(..., description="The full text of the research paper")


# Response Models
class UploadResponse(BaseModel):
    """Response model for PDF upload and analysis"""

    success: bool
    filename: str
    text: str
    summary: str
    message: str


class QuestionResponse(BaseModel):
    """Response model for question answering"""

    success: bool
    response: str


class CodeResponse(BaseModel):
    """Response model for code generation"""

    success: bool
    code: str


class AnalysisResponse(BaseModel):
    """Response model for paragraph analysis"""

    success: bool
    analysis: str


class TechnicalAnalysisResponse(BaseModel):
    """Response model for technical analysis"""

    success: bool
    technical_analysis: str


class ErrorResponse(BaseModel):
    """Error response model"""

    detail: str
    error_code: Optional[str] = None
