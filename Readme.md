# MedAssist AI RAG - Medical Chatbot

## 🏥 Overview

MedAssist AI RAG is a production-ready medical customer service chatbot powered by Retrieval-Augmented Generation (RAG). It combines vector search with large language models to provide accurate, context-aware medical information while maintaining strict safety protocols.

## 🧠 Architecture

### RAG Pipeline
1. **Query Processing** - Emergency detection and intent analysis
2. **Vector Retrieval** - Semantic search in medical knowledge base
3. **LLM Generation** - Context-aware response generation using Gemini
4. **Safety Validation** - Medical disclaimers and source citations

### Technology Stack
- **Frontend**: Streamlit
- **Vector Database**: FAISS
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini (gemini-1.5-flash)
- **Database**: SQLite
- **Deployment**: Docker + Docker Compose

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Google AI API key (for Gemini)

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd medassist-ai-rag
cp .env.example .env
```

2. **Configure environment**:
Edit `.env` file and add your Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```

3. **Deploy**:
```bash
chmod +x deploy.sh
./deploy.sh
```

4. **Access**:
- Application: http://localhost:8501
- Monitoring: http://localhost:3000 (optional)

## 🔧 Configuration

### Core Settings
```bash
# LLM Configuration
GEMINI_MODEL=gemini-1.5-flash
MAX_TOKENS=1500
TEMPERATURE=0.3

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7
```

### Rate Limiting
```bash
RATE_LIMIT_REQUESTS=100  # Requests per window
RATE_LIMIT_WINDOW=3600   # Window in seconds
```

## 🏗️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

### Docker Development
```bash
# Build
make build

# Run
make run

# View logs
make logs

# Stop
make stop
```

## 📊 Features

### Core Functionality
- ✅ RAG-powered medical Q&A
- ✅ Emergency situation detection
- ✅ Vector semantic search
- ✅ LLM response generation
- ✅ Conversation persistence
- ✅ Source citation
- ✅ Confidence scoring

### Safety Features
- ✅ Medical disclaimers
- ✅ Emergency routing
- ✅ Rate limiting
- ✅ Input validation
- ✅ Audit logging

### Production Features
- ✅ Docker containerization
- ✅ Health checks
- ✅ Monitoring ready
- ✅ Database persistence
- ✅ Error handling
- ✅ Graceful degradation

## 🩺 Medical Knowledge Base

The system includes comprehensive medical information covering:

- **Respiratory Conditions** - Symptoms, COVID-19, treatment
- **Medication Safety** - Drug interactions, dosing, storage
- **Emergency Recognition** - Critical symptoms, when to call 911
- **Preventive Healthcare** - Screenings, vaccinations, lifestyle
- **Healthcare Navigation** - Appointment scheduling, insurance
- **Mental Health** - Resources, crisis support, treatment options

## 🔒 Privacy & Security

- Conversations stored locally only
- No PHI transmitted to external services
- Rate limiting protection
- Input sanitization
- Secure containerized deployment

## 📈 Monitoring

Optional monitoring stack includes:
- Prometheus for metrics collection
- Grafana for visualization