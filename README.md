# UDOM RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) system tailored for the **University of Dodoma (UDOM)**. The system combines multi-provider Large Language Models (LLMs), hybrid dense/sparse vector retrieval, intent-aware query routing, automated site crawling, and a responsive React frontend to deliver accurate, source-backed answers for academic, administrative, and general institutional inquiries.

---

## 🌟 Key Features

### 🚀 Advanced RAG & Retrieval Engine
- **Hybrid Search**: Fuses dense vector search ([`pgvector`](https://github.com/pgvector/pgvector) with `bge-m3` embeddings) and sparse Full-Text Search (PostgreSQL `tsvector`) using **Reciprocal Rank Fusion (RRF)**.
- **Intent Routing & Specialized Pipelines**: Automatically routes queries based on domain intent:
  - 📅 **Almanac & Academic Calendar**: Key academic dates, deadlines, and semester schedules.
  - 📚 **Curriculum & Courses**: Programme requirements, course codes, and unit distributions.
  - 📢 **Announcements & Crawled Web Content**: Timely news and official university updates.
  - ❓ **FAQ Matching**: Direct high-confidence lookup against indexed FAQs.
  - 📝 **Procedural & Institutional Guidance**: Step-by-step guidance for administrative procedures.
- **Reranking & Grounding**: Integrates Jina Reranker / local cross-encoders with adaptive thresholds, citation verification, and automated answer repair to prevent hallucinations.
- **Multilingual & Alias Expansion**: English and Swahili terminology normalization and semantic alias mapping.

### 🛡️ Multi-Provider LLM & Resilience
- **Multi-Model Support**: Seamless integration with **Groq**, **Google Gemini**, **Ollama**, and **OpenAI**.
- **Circuit Breakers & Automatic Failover**: Fallback chains across model providers to guarantee uptime during rate limits or service disruptions.
- **Context Compaction & Anaphora Resolution**: Dynamic memory optimization for long multi-turn conversations.

### 💬 Interactive Real-Time Frontend
- **Modern UI**: Built with **React 19**, **Vite**, and **TailwindCSS**, featuring dark/light mode toggle.
- **Streaming Responses**: Real-time response generation via **WebSockets** and **Server-Sent Events (SSE)**.
- **Citation & Source Transparency**: Displays verifiable document sources, page numbers, and downloadable attachments.
- **Session Management & Feedback**: Conversation history, guest/authenticated sessions, and message feedback (Thumbs up/down).

### 🔧 Comprehensive Admin & Ingestion Portal
- **Document Ingestion**: Supports PDF, DOCX, TXT, CSV, and XLSX parsing with automatic chunking, metadata extraction, and quality verification.
- **Automated Web Crawler**: Configurable background crawler for full site indexing and automated announcement tracking.
- **System Control & Audit Logging**: Dynamic runtime system settings management with complete admin action auditing.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend (React + Vite)"]
        UI[User Interface / Chat]
        WS[WebSocket / SSE Stream]
    end

    subgraph API ["FastAPI Gateway"]
        Auth[Auth & Security Middleware]
        Limiter[Rate Limiter - Redis / Local]
        Routes[API Routes: Chat, Query, Admin]
    end

    subgraph RAG ["RAG Processing Engine"]
        Intent[Intent Router & Normalizer]
        Hybrid[Hybrid Retriever: Dense + Sparse]
        RRF[Reciprocal Rank Fusion]
        Rerank[Reranker Service - Jina / Cross-Encoder]
        Gen[Generation & Resilience Service]
    end

    subgraph LLM ["LLM Providers"]
        Groq[Groq / Llama 3]
        Gemini[Google Gemini]
        Ollama[Ollama / Local LLM]
    end

    subgraph Data ["Storage & Database"]
        PG[(PostgreSQL + pgvector)]
        Redis[(Redis Cache / Rate Limiting)]
        Files[Uploaded Documents / Static Assets]
    end

    UI --> WS
    WS --> Routes
    Routes --> Auth --> Limiter
    Limiter --> Intent
    Intent --> Hybrid
    Hybrid --> PG
    PG --> RRF
    RRF --> Rerank
    Rerank --> Gen
    Gen --> Groq & Gemini & Ollama
    Gen --> Routes --> UI
```

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/), Python 3.10+, Uvicorn |
| **Database** | [PostgreSQL](https://www.postgresql.org/) + [`pgvector`](https://github.com/pgvector/pgvector), [SQLAlchemy 2.0](https://www.sqlalchemy.org/), [Alembic](https://alembic.sqlalchemy.org/) |
| **AI / Embeddings** | Ollama (`bge-m3`), LangChain Core, Jina Reranker API, Groq SDK, Google Gemini |
| **Caching & Limiting** | Redis, In-memory Leaky Bucket Rate Limiter |
| **Parsing & Crawling** | `pdfplumber`, `pypdf`, `python-docx`, `beautifulsoup4`, `httpx`, `markdownify` |
| **Frontend** | [React 19](https://react.dev/), [Vite](https://vitejs.dev/), [TailwindCSS v4](https://tailwindcss.com/), `react-markdown`, `remark-gfm` |
| **Task Scheduling** | APScheduler |

---

## 📁 Directory Structure

```
rag-chatbot/
├── app/
│   ├── api/
│   │   ├── routes/         # API endpoints (admin, auth, chat, query, ws_chat)
│   │   ├── deps.py         # FastApi dependency injection & authentication
│   │   └── limiter.py      # Rate limiting logic (Redis / local fallback)
│   ├── core/
│   │   ├── config.py       # Pydantic Settings & system configurations
│   │   ├── security.py     # Password hashing & JWT validation
│   │   └── scheduler.py    # Background cron tasks (crawler, file reconciliation)
│   ├── db/
│   │   ├── models.py       # SQLAlchemy ORM models (Users, Documents, Chunks, FAQs, etc.)
│   │   └── session.py      # PostgreSQL database session management
│   ├── services/           # Core RAG pipeline, intent router, hybrid retriever, generator
│   ├── main.py             # FastAPI entrypoint & application setup
│   └── schemas/            # Pydantic request/response validation schemas
├── alembic/                # Database migrations
├── rag-frontend/           # React + Vite frontend application
│   ├── src/                # Components, pages, hooks, styling
│   └── package.json        # Node.js dependencies & scripts
├── scripts/                # Utility scripts (create_admin.py, import_retrieval_aliases.py, etc.)
├── tests/                  # Unit and integration test suites
│   ├── unit/               # Detailed service & logic unit tests
│   └── integration/        # Pipeline & route integration tests
├── .env.example            # Environment variable template
├── requirements.txt        # Python backend dependencies
└── pytest.ini              # Pytest configuration
```

---

## ⚙️ Getting Started

### Prerequisites

- **Python**: `3.10` or higher
- **Node.js**: `18.x` or higher (`npm` included)
- **PostgreSQL**: `15+` with the `pgvector` extension enabled.
- **Ollama** *(Optional for local embeddings/models)*: Download from [ollama.ai](https://ollama.ai/)

---

### 1. Database Setup

1. Create a PostgreSQL database (e.g., `rag_db`):
   ```sql
   CREATE DATABASE rag_db;
   \c rag_db;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

---

### 2. Backend Setup

1. **Clone repository & set up Python virtual environment**:
   ```bash
   git clone https://github.com/your-repo/rag-chatbot.git
   cd rag-chatbot

   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your database credentials and API keys:
   ```bash
   cp .env.example .env
   ```

   *Key Configuration Options in `.env`:*
   ```env
   DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
   AUTH_SECRET_KEY=your-super-secret-jwt-key
   
   # Embedding Configuration
   EMBEDDING_PROVIDER=ollama
   EMBEDDING_MODEL=bge-m3
   EMBEDDING_DIMENSION=1024
   OLLAMA_BASE_URL=http://localhost:11434

   # LLM Provider API Keys
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   JINA_API_KEY=your_jina_api_key
   ```

4. **Run Database Migrations**:
   ```bash
   alembic upgrade head
   ```

5. **Create Admin User**:
   ```bash
   python scripts/create_admin.py --username admin --password YourSecurePassword
   ```

6. **Start Backend Server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   The interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### 3. Frontend Setup

1. **Navigate to `rag-frontend` directory**:
   ```bash
   cd rag-frontend
   ```

2. **Install Node dependencies**:
   ```bash
   npm install
   ```

3. **Run Development Server**:
   ```bash
   npm run dev
   ```
   Access the web application at [http://localhost:5173](http://localhost:5173).

---

## 🧪 Running Tests

The test suite includes extensive unit and integration tests covering intent routing, hybrid search, grounding, failover logic, and document ingestion.

Run pytest from the project root:
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run tests with coverage summary
pytest --cov=app
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness & readiness health probe (DB & Rate Limiter) |
| `POST` | `/api/auth/login` | User authentication & JWT issuance |
| `POST` | `/api/chat` | Non-streaming / SSE streaming chat endpoint |
| `WS` | `/ws/chat` | Real-time WebSocket streaming chat connection |
| `POST` | `/api/query` | Direct query/retrieval search endpoint |
| `POST` | `/api/admin/documents` | Upload and process new document for indexing |
| `GET` | `/api/admin/documents` | List indexed documents with quality & indexing status |
| `POST` | `/api/admin/crawler/start` | Trigger automated website background crawler |
| `GET` | `/api/admin/system-settings` | Manage dynamic system configuration settings |

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.