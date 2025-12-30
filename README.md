# Nutrition Intelligence Backend

Evidence-based nutrition recommendations powered by RAG (Retrieval-Augmented Generation) with clinical guidelines.

## Features

- FastAPI backend with structured nutrition targets
- RAG-powered recommendations using Ollama (llama3.2)
- Qdrant vector database for clinical guidelines (ADA, AHA, DASH, KDIGO)
- TDEE calculation with Mifflin-St Jeor formula
- Kidney function safety checks (eGFR-based protein restriction)

## Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Run the application
python -m app.main
```

## API Endpoints

- `POST /recommend` - Text-based nutrition recommendations
- `POST /nutrition/targets` - Structured nutrition targets
- `GET /health` - System health check
- `GET /docs` - Interactive API documentation

## Requirements

- Python 3.11+
- Ollama with llama3.2 model
- Qdrant vector database

## License

MIT License - See LICENSE file

## Disclaimer

This software is for educational purposes only. Always consult with a qualified healthcare provider before making dietary changes.
