# Smart CV Screening & Matching System

Comprehensive AI-powered system for intelligent Recruitment that automates CV parsing, screening, and matching. Built with advanced RAG (Retrieval-Augmented Generation) and Agentic Workflow.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)

## Key Features

### Intelligent Parsing & Screening
- **Multi-Modal Ingestion**: Handles PDFs, Images, and scanned documents using **PaddleOCR**.
- **Vietnamese NLP**: Specialized support for Vietnamese CVs using **Underthesea** for word segmentation.
- **LLM-Powered Extraction**: Uses **GPT-4o** with Structured Outputs to extract entities (Education, Experience, Skills) with high precision.
- **Auto-Correction**: **LLM Evaluator** checks extracted data quality and auto-reformats inconsistencies.

### Advanced Search & Matching
- **Hybrid Search**: Combines **BM25** (keyword) and **Vector Search** (semantic) using **Reciprocal Rank Fusion (RRF)**.
- **Semantic Understanding**: Uses `BAAI/bge-m3` embedding model for deep understanding of technical skills and job descriptions.
- **Smart Matching**: Auto-calculates years of experience (handling overlapping timelines) and matches candidates to JD.

### Batch Processing & Integration
- **Batch Import**: Scan and process entire directories of CVs in one click.
- **Google Drive Integration**: Import CVs directly from public Google Drive folders.
- **Concurrency Control**: Optimized for stability with managed Celery workers (configurable concurrency).

### RAG Chat Assistant
- **Context-Aware Chat**: Chat with your candidate database. Ask questions like *"Who knows Python and has >3 years exp?"*.
- **Grounded Responses**: Answers are strictly based on CV data, cited with sources.
- **Zero-Latency Response**: Optimized streaming pipeline for instant feedback.

---

## System Architecture

### Processing Pipeline (Ingestion)
The system transforms raw unstructured data into a structured knowledge base using a multi-stage AI pipeline.

```mermaid
flowchart TD
    %% Premium Color Palette
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c;
    classDef ai fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100;
    classDef storage fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#1b5e20;
    
    subgraph Input_Layer [Input Layer]
        direction TB
        A(Upload CV):::input --> B{Type Check?}
    end

    subgraph Extraction_Layer [Bio-Extraction Layer]
        direction TB
        B -->|Image/Scan| C[PaddleOCR]:::process
        B -->|Text PDF| D[PyPDF/Miner]:::process
    end

    subgraph Intelligence_Layer [AI Core Layer]
        direction TB
        C & D --> E[GPT-4o Parser]:::ai
        E --> F{Quality Check}
        F -->|Fail| G[LLM Reformatter]:::ai
        G --> H[BAAI/bge-m3 Embedding]:::ai
        F -->|Pass| H
    end

    subgraph Storage_Layer [Knowledge Base]
        direction TB
        H --> I[(PgVector DB)]:::storage
        H --> J[(BM25 Index)]:::storage
    end

    Input_Layer --> Extraction_Layer
    Extraction_Layer --> Intelligence_Layer
    Intelligence_Layer --> Storage_Layer
```

**Workflow Technologies:**
*   **Input**: Auto-sensing file type detection.
*   **OCR Engine**: `PaddleOCR` (Optimized for Vietnamese/English).
*   **Parsing Engine**: `OpenAI GPT-4o` (Structured Output Mode).
*   **Embedding Model**: `BAAI/bge-m3` (Multi-lingual, 1024 dim).
*   **Database**: `PostgreSQL 16` + `pgvector` extension.

### RAG Chat Pipeline (Retrieval & Generation)
An Agentic RAG workflow delivering grounded, accurate answers.

```mermaid
flowchart TD
    %% Premium Color Palette
    classDef user fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#0277bd;
    classDef logic fill:#fff8e1,stroke:#ff8f00,stroke-width:2px,color:#ff8f00;
    classDef search fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#33691e;
    classDef gen fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2;

    User(User Query):::user --> Planner[Query Planner]:::logic
    
    subgraph Search_Engine [Hybrid Search Engine]
        direction TB
        Planner -->|Expanded Terms| V[Hybrid Search Controller]:::search
        V -->|Semantic| VS[(PgVector)]
        V -->|Keyword| KS[(BM25)]
        VS & KS --> RRF[RRF Fusion]:::search
    end

    subgraph Generation_Layer [Generation Layer]
        direction TB
        RRF --> Context[Candidate Context]
        Context --> Gen[GPT-4o Generator]:::gen
        Gen --> Critic{Response Critic}:::gen
        Critic -->|Retry| Gen
        Critic -->|Approved| Final[Final Response Input]:::user
    end

    Planner --> Search_Engine
    Search_Engine --> Generation_Layer
```

**Pipeline Technologies:**
*   **Query Expansion**: `GPT-4o` rephrases queries for better recall.
*   **Hybrid Search**:
    *   **Semantic**: Cosine Similarity via `pgvector`.
    *   **Keyword**: `BM25` algorithm for exact matches.
    *   **Fusion**: Reciprocal Rank Fusion (**RRF**) algorithm.
*   **Generation**: `GPT-4o` with strict context grounding.
*   **Quality Control**: Secondary LLM "Critic" evaluates answer accuracy before display.

---

## Installation & Setup

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
   You can run the project directly with Docker (Recommended):

   ```bash
   # Option A: Quick Start script
   ./start_project.sh

   # Option B: Manual Docker Command
   # Build and start services (First time may take 5-10 mins to download AI models)
   docker-compose up --build -d
   ```

   *The system will be available at:*
   - **Frontend**: [http://localhost:3000](http://localhost:3000)
   - **Backend API**: [http://localhost:8000](http://localhost:8000)
   - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Flower (Worker Monitor)**: [http://localhost:5555](http://localhost:5555)

4. **Common Docker Commands**:
   ```bash
   # View logs (real-time)
   docker-compose logs -f

   # Stop all services
   docker-compose down

   # Stop and remove volumes (Caution: Deletes DB data)
   docker-compose down -v
   ```

5. **Stop the System**:
   ```bash
   ./stop_project.sh
   # OR
   docker-compose down
   ```

---

## Usage Guide

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

## Project Structure

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

## Contribution

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

