import PyPDF2
import io
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFService:
    """Service for handling PDF operations"""

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text from the PDF

        Raises:
            Exception: If PDF processing fails
        """
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

                if not text.strip():
                    raise Exception("No text could be extracted from the PDF")

                return text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    async def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes

        Args:
            pdf_bytes: PDF file as bytes

        Returns:
            Extracted text from the PDF
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""

            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"

            if not text.strip():
                raise Exception("No text could be extracted from the PDF")

            return text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def validate_pdf(self, file_path: str) -> bool:
        """
        Validate if the file is a valid PDF

        Args:
            file_path: Path to the file

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            with open(file_path, "rb") as file:
                PyPDF2.PdfReader(file)
                return True
        except Exception:
            return False
