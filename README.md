# Smart CV Screening & Matching System 🚀

Comprehensive AI-powered system for intelligent Recruitment that automates CV parsing, screening, and matching. Built with advanced RAG (Retrieval-Augmented Generation) and Agentic Workflow.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)

## ✨ Key Features

### 🧠 Intelligent Parsing & Screening
- **Multi-Modal Ingestion**: Handles PDFs, Images, and scanned documents using **PaddleOCR**.
- **Vietnamese NLP**: Specialized support for Vietnamese CVs using **Underthesea** for word segmentation.
- **LLM-Powered Extraction**: Uses **GPT-4o** with Structured Outputs to extract entities (Education, Experience, Skills) with high precision.
- **Auto-Correction**: **LLM Evaluator** checks extracted data quality and auto-reformats inconsistencies.

### 🔍 Advanced Search & Matching
- **Hybrid Search**: Combines **BM25** (keyword) and **Vector Search** (semantic) using **Reciprocal Rank Fusion (RRF)**.
- **Semantic Understanding**: Uses `BAAI/bge-m3` embedding model for deep understanding of technical skills and job descriptions.
- **Smart Matching**: Auto-calculates years of experience (handling overlapping timelines) and matches candidates to JD.

### ⚡ Batch Processing & Integration
- **Batch Import**: Scan and process entire directories of CVs in one click.
- **Google Drive Integration**: Import CVs directly from public Google Drive folders.
- **Concurrency Control**: Optimized for stability with managed Celery workers (configurable concurrency).

### 💬 RAG Chat Assistant
- **Context-Aware Chat**: Chat with your candidate database. Ask questions like *"Who knows Python and has >3 years exp?"*.
- **Grounded Responses**: Answers are strictly based on CV data, cited with sources.
- **Zero-Latency Response**: Optimized streaming pipeline for instant feedback.

---

## 🏗️ System Architecture

### Processing Pipeline
```mermaid
flowchart LR
    A[Upload CV] --> B{Type?}
    B -->|PDF/Image| C[PaddleOCR]
    B -->|Text| D[Text Extract]
    C & D --> E[LLM Parser]
    E --> F[Quality Check]
    F -->|Fail| G[Reformat]
    F -->|Pass| H[Embedding]
    G --> H
    H --> I[(Vector DB)]
    H --> J[(Keyword Index)]
```

### RAG Chat Pipeline
```mermaid
flowchart LR
    User[User Query] --> Planner[Query Planner]
    Planner -->|Search| Hybrid[Hybrid Search]
    Hybrid -->|Retrieve| Context[Candidate Data]
    Context --> Critic[Response Generator]
    Critic -->|Evaluate| Final[Final Response]
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Docker & Docker Compose**
- **Python 3.11+** (for local scripts)
- **Node.js 18+** (for frontend development)
- **OpenAI API Key**

### Quick Start (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd smart-cv-screening
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Open .env and add your OPENAI_API_KEY
   ```

3. **Start the System**:
   Use the helper script to set up everything (Virtual Env, Docker, Backend, Frontend):
   ```bash
   ./start_project.sh
   ```
   *The system will be available at:*
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

4. **Stop the System**:
   ```bash
   ./stop_project.sh
   ```

---

## 📖 Usage Guide

### 1. Uploading CVs
- **Single Upload**: Drag & drop a file on the Chat Interface.
- **Batch Import**: 
  - Click the **"Batch Import"** button in the header.
  - **Local Folder**: Enter server path (e.g., `./public_cvs`).
  - **Google Drive**: Paste a public folder link (e.g., `https://drive.google.com/...`).

### 2. Chatting & Searching
- Type natural language queries:
  - *"Find Java developers with Spring Boot experience."*
  - *"Compare Candidate A and Candidate B."*
  - *"Who is the best fit for a Senior Frontend role?"*

### 3. CLI Tools
The project includes helpful CLI scripts:
- `python3 list_db.py`: View all candidates in the database.
- `python3 scan_cvs.py <directory>`: Trigger batch processing from CLI.
- `python3 clear_db.py`: **(Caution)** Wipe the database.

---

## 📦 Project Structure

```text
smart-cv-screening/
├── app/                    # Backend Source Code
│   ├── api/                # FastAPI Routes
│   ├── core/               # Config & Celery
│   ├── models/             # Database Models
│   ├── schemas/            # Pydantic Schemas
│   ├── services/           # Core Logic
│   │   ├── ingestion/      # OCR & Preprocessing
│   │   ├── parsing/        # LLM Parsing & Evaluation
│   │   ├── embedding/      # Vector Generation
│   │   ├── search/         # Search Engines
│   │   └── chat/           # RAG Pipeline
│   └── workers/            # Background Tasks
├── frontend/               # Next.js Frontend
│   ├── app/                # Pages & Layouts
│   ├── components/         # UI Components
│   └── lib/                # API Clients
├── scripts/                # Utility Scripts
├── .env.example            # Environment Template
├── docker-compose.yml      # Infrastructure
├── requirements.txt        # Python Dependencies
├── start_project.sh        # Startup Script
└── stop_project.sh         # Shutdown Script
```

## 🤝 Contribution

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
