# 🍎 Nutrition Intelligence Backend

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Evidence-based nutrition recommendations powered by RAG (Retrieval-Augmented Generation) with clinical guidelines.

## ✨ Features

- 🚀 **FastAPI Backend**: High-performance async API
- 🧠 **RAG-Powered**: Retrieves from clinical guidelines (ADA, AHA, DASH, KDIGO)
- 🤖 **Local LLM**: Ollama (llama3.2) for privacy-first inference
- 📊 **Structured Targets**: Daily nutrition targets with TDEE calculation
- ⚕️ **Safety Checks**: Kidney function monitoring (eGFR-based protein restriction)
- 🔍 **Vector Search**: Qdrant for semantic clinical guideline retrieval

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) with llama3.2 model
- [Qdrant](https://qdrant.tech/) vector database

### Installation
```bash
# Clone repository
git clone https://github.com/seanqxu/nutrition-rag-backend.git
cd nutrition-rag-backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Run application
python -m app.main
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/recommend` | POST | Text-based nutrition recommendations |
| `/nutrition/targets` | POST | Structured nutrition targets with TDEE |
| `/docs` | GET | Interactive API documentation |

## 🏗️ Architecture
```
┌─────────────────────────────────────────┐
│         FastAPI Application              │
│  ┌───────────────────────────────────┐  │
│  │    Nutrition Mapper Module        │  │
│  │  - TDEE Calculation               │  │
│  │  - Safety Checks (eGFR)           │  │
│  └─────────────┬─────────────────────┘  │
│                │                         │
│  ┌─────────────▼─────────────────────┐  │
│  │         RAG Engine                │  │
│  │  - Embedder (nomic-embed-text)    │  │
│  │  - Retriever (Qdrant search)      │  │
│  │  - Generator (Ollama LLM)         │  │
│  └────────┬────────────┬──────────────┘  │
└───────────┼────────────┼─────────────────┘
            │            │
            ▼            ▼
     ┌──────────┐  ┌──────────┐
     │  Qdrant  │  │  Ollama  │
     │  Vector  │  │   LLM    │
     │   Store  │  │ (llama3) │
     └──────────┘  └──────────┘
```

## 🔧 Configuration

Edit `.env` file:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
QDRANT_HOST=localhost
QDRANT_PORT=6333
API_PORT=8000
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## ⚠️ Disclaimer

This software is for educational purposes only. Always consult with a qualified healthcare provider before making dietary changes.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Built with ❤️ for evidence-based nutrition**
