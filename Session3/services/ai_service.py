import google.generativeai as genai
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AIService:
    """Service for AI operations using Google Gemini"""

    def __init__(self):
        """Initialize the AI service with Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    async def generate_summary(self, paper_text: str) -> str:
        """
        Generate a comprehensive summary of the research paper

        Args:
            paper_text: The full text of the research paper

        Returns:
            AI-generated summary in layman's terms
        """
        try:
            prompt = f"""Please provide a comprehensive summary of this research paper in simple, layman's terms. Include:
1. Main research question/problem
2. Key findings
3. Methodology used
4. Significance and implications
5. Technical terms explained in simple language

Make it easy for someone without technical background to understand.

Research Paper Content:
{paper_text}"""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise Exception(f"Failed to generate summary: {str(e)}")

    async def answer_question(self, question: str, paper_text: str) -> str:
        """
        Answer a specific question about the research paper

        Args:
            question: The question to answer
            paper_text: The full text of the research paper

        Returns:
            AI-generated answer
        """
        try:
            prompt = f"""Research Paper Content:
{paper_text}

User Question: {question}

Please provide a comprehensive and detailed response that directly addresses the question using information from the research paper."""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise Exception(f"Failed to answer question: {str(e)}")

    async def generate_code(
        self, algorithm: str, language: str, paper_text: str
    ) -> str:
        """
        Generate code implementation based on the research paper

        Args:
            algorithm: Description of the algorithm to implement
            language: Programming language
            paper_text: The full text of the research paper

        Returns:
            Generated code with explanations
        """
        try:
            prompt = f"""Based on the research paper content and the algorithm description, please generate {language} code implementation.

Research Paper Content:
{paper_text}

Algorithm: {algorithm}
Programming Language: {language}

Please provide:
1. Complete, runnable code
2. Comments explaining each major section
3. Example usage
4. Time and space complexity analysis
5. Any important notes or considerations

Make sure the code is well-structured and follows best practices for {language}."""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise Exception(f"Failed to generate code: {str(e)}")

    async def analyze_paragraph(self, paragraph: str, paper_text: str) -> str:
        """
        Analyze a specific paragraph from the research paper

        Args:
            paragraph: The paragraph text to analyze
            paper_text: The full text of the research paper

        Returns:
            Detailed analysis of the paragraph
        """
        try:
            prompt = f"""Please analyze this specific paragraph from the research paper:

Research Paper Content:
{paper_text}

Paragraph to analyze: "{paragraph}"

Please provide:
1. Simple explanation of what this paragraph means
2. Key concepts and terms explained in layman's terms
3. How this paragraph relates to the overall research
4. Any important implications or insights
5. Technical details broken down into simple language

Make it accessible to someone without a technical background."""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error analyzing paragraph: {str(e)}")
            raise Exception(f"Failed to analyze paragraph: {str(e)}")

    async def get_technical_analysis(self, paper_text: str) -> str:
        """
        Get detailed technical analysis of the research paper

        Args:
            paper_text: The full text of the research paper

        Returns:
            Comprehensive technical analysis
        """
        try:
            prompt = f"""Please provide a detailed technical analysis of this research paper:

Research Paper Content:
{paper_text}

Please provide:
1. **Methodology Deep Dive**: Explain the research methods, algorithms, and techniques used
2. **Mathematical Foundations**: Break down any mathematical concepts and formulas
3. **Experimental Design**: Analyze the experimental setup, data collection, and validation methods
4. **Results Interpretation**: Detailed analysis of the results and their statistical significance
5. **Limitations and Future Work**: Discuss the paper's limitations and potential improvements
6. **Code Implementation Insights**: Provide guidance on how to implement the key algorithms
7. **Comparison with Related Work**: How this research compares to existing literature

Please be thorough and technical while still being accessible."""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating technical analysis: {str(e)}")
            raise Exception(f"Failed to generate technical analysis: {str(e)}")
