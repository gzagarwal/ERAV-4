# Research Paper Analyzer

A beautiful, AI-powered web application for analyzing research papers using Google's Gemini API. Upload PDF research papers and get instant AI-generated summaries, ask questions, generate code implementations, and get detailed technical analysis.

## Features

- 📄 **PDF Upload & Analysis**: Upload research papers and get instant AI-powered analysis
- 🔍 **Comprehensive Summaries**: Get layman-friendly summaries with key findings and methodology
- ❓ **Interactive Q&A**: Ask specific questions about the paper and get detailed answers
- 💻 **Code Generation**: Generate code implementations in multiple languages (Python, Java, Go, etc.)
- 📝 **Paragraph Analysis**: Analyze specific paragraphs for detailed explanations
- 🔬 **Technical Deep Dive**: Get comprehensive technical analysis including methodology, math foundations, and implementation insights
- 🎨 **Beautiful UI**: Modern, responsive design with smooth animations
- 🚀 **FastAPI Backend**: High-performance Python backend with automatic API documentation

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Google Gemini AI**: Advanced AI for paper analysis and code generation
- **PyPDF2**: PDF text extraction
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: Modern React with hooks
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Smooth animations and transitions
- **React Markdown**: Markdown rendering
- **React Syntax Highlighter**: Code syntax highlighting
- **Axios**: HTTP client
- **React Hot Toast**: Beautiful notifications

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+
- Google Gemini API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd research-paper-analyzer
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Frontend Setup
```bash
cd client
npm install
cd ..
```

### 4. Run the Application

#### Development Mode
```bash
# Terminal 1: Start FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start React frontend
cd client
npm start
```

#### Production Mode
```bash
# Build frontend
cd client
npm run build
cd ..

# Start backend (serves both API and frontend)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Docker Deployment

### Using Docker Compose
```bash
# Set up environment variables
cp env.example .env
# Edit .env and add your GEMINI_API_KEY

# Build and run
docker-compose up --build

# For production with nginx
docker-compose --profile production up --build
```

### Using Docker
```bash
# Build image
docker build -t research-analyzer .

# Run container
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here research-analyzer
```

## API Endpoints

### Core Endpoints
- `POST /api/upload-pdf` - Upload and analyze PDF
- `POST /api/ask-question` - Ask questions about the paper
- `POST /api/generate-code` - Generate code implementations
- `POST /api/analyze-paragraph` - Analyze specific paragraphs
- `POST /api/technical-analysis` - Get technical deep dive analysis
- `GET /api/health` - Health check

### Interactive Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
PORT=8000
```

### Getting Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Usage Examples

### Upload and Analyze a Paper
1. Navigate to the web interface
2. Drag and drop a PDF research paper
3. Wait for AI analysis to complete
4. View the generated summary

### Ask Questions
1. Go to the "Ask Questions" tab
2. Use predefined questions or ask your own
3. Get instant AI-powered answers

### Generate Code
1. Go to the "Generate Code" tab
2. Select programming language
3. Describe the algorithm to implement
4. Get complete, runnable code with explanations

### Analyze Paragraphs
1. Go to the "Analyze Paragraph" tab
2. Select a paragraph from the paper or paste your own
3. Get detailed analysis and explanations

## Features in Detail

### AI-Powered Analysis
- **Layman Summaries**: Complex research explained in simple terms
- **Technical Deep Dives**: Comprehensive analysis for experts
- **Code Generation**: Multiple programming languages supported
- **Interactive Q&A**: Natural language question answering

### Beautiful User Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Framer Motion powered transitions
- **Modern Styling**: Tailwind CSS with custom color schemes
- **Accessibility**: WCAG compliant design

### Developer Experience
- **Type Safety**: Pydantic models for request/response validation
- **Auto Documentation**: FastAPI generates OpenAPI specs
- **Hot Reload**: Development server with automatic reloading
- **Error Handling**: Comprehensive error handling and user feedback

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Google Gemini AI for powerful language understanding
- FastAPI for the excellent web framework
- React and Tailwind CSS for the beautiful frontend
- The open-source community for various libraries and tools
