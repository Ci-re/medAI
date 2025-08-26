# app.py - Production Medical RAG Chatbot with LLM Integration
import streamlit as st
import os
import logging
from datetime import datetime
import json
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from urllib.parse import urlparse
import requests
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class Config:
    APP_NAME: str = "MedAssist AI RAG"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = "data/conversations.db"
    VECTOR_DB_PATH: str = "data/vector_store"
    LOG_LEVEL: str = "INFO"
    MAX_CONVERSATION_LENGTH: int = 50
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600

    GEMINI_API_KEY: str = "AIzaSyAz5XgxgvzTEO8RQcNrfUsO-u_66ku6394"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.3

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.4
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    @classmethod
    def from_env(cls):
        return cls(
            APP_NAME=os.getenv("APP_NAME", cls.APP_NAME),
            VERSION=os.getenv("APP_VERSION", cls.VERSION),
            DATABASE_URL=os.getenv("DATABASE_URL", cls.DATABASE_URL),
            VECTOR_DB_PATH=os.getenv("VECTOR_DB_PATH", cls.VECTOR_DB_PATH),
            LOG_LEVEL=os.getenv("LOG_LEVEL", cls.LOG_LEVEL),
            MAX_CONVERSATION_LENGTH=int(
                os.getenv("MAX_CONVERSATION_LENGTH", cls.MAX_CONVERSATION_LENGTH)
            ),
            RATE_LIMIT_REQUESTS=int(
                os.getenv("RATE_LIMIT_REQUESTS", cls.RATE_LIMIT_REQUESTS)
            ),
            RATE_LIMIT_WINDOW=int(
                os.getenv("RATE_LIMIT_WINDOW", cls.RATE_LIMIT_WINDOW)
            ),
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            GEMINI_MODEL=os.getenv("GEMINI_MODEL", cls.GEMINI_MODEL),
            MAX_TOKENS=int(os.getenv("MAX_TOKENS", cls.MAX_TOKENS)),
            TEMPERATURE=float(os.getenv("TEMPERATURE", cls.TEMPERATURE)),
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", cls.EMBEDDING_MODEL),
            TOP_K_RETRIEVAL=int(os.getenv("TOP_K_RETRIEVAL", cls.TOP_K_RETRIEVAL)),
            SIMILARITY_THRESHOLD=float(
                os.getenv("SIMILARITY_THRESHOLD", cls.SIMILARITY_THRESHOLD)
            ),
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", cls.CHUNK_SIZE)),
            CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", cls.CHUNK_OVERLAP)),
        )


config = Config.from_env()

# Initialize Gemini
if config.GEMINI_API_KEY:
    print(config.GEMINI_API_KEY)
    genai.configure(api_key=config.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not configured. Using mock responses for demo.")

# Medical Knowledge Base - Enhanced for RAG
MEDICAL_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "respiratory_conditions",
        "title": "Respiratory Conditions and Symptoms",
        "content": """
        Respiratory conditions affect the lungs and breathing passages. Common symptoms include:
        
        Cough: Can be dry (non-productive) or wet (productive with phlegm). Persistent cough lasting more than 2 weeks should be evaluated by a healthcare provider.
        
        Shortness of breath (dyspnea): Difficulty breathing during normal activities or at rest. Can indicate various conditions from asthma to heart disease.
        
        Wheezing: High-pitched whistling sound when breathing, often associated with asthma or COPD.
        
        Chest tightness: Feeling of pressure or squeezing in the chest area.
        
        COVID-19 symptoms: Fever, dry cough, fatigue, loss of taste or smell, body aches, sore throat, congestion.
        
        When to seek care: Difficulty breathing, chest pain, high fever, coughing up blood, or symptoms that worsen rapidly.
        
        Prevention: Hand hygiene, masks in crowded areas, vaccinations, avoiding smoking and air pollutants.
        """,
        "category": "medical_conditions",
        "source": "CDC Guidelines",
        "last_updated": "2024-01-15",
    },
    {
        "id": "medication_safety",
        "title": "Medication Safety and Drug Interactions",
        "content": """
        Medication safety is critical for effective treatment and avoiding adverse effects.
        
        Drug interactions occur when medications affect each other's effectiveness or increase side effects. Types include:
        - Drug-drug interactions: Between prescription medications
        - Drug-food interactions: Medications affected by certain foods
        - Drug-supplement interactions: Prescription drugs with vitamins or herbs
        
        Important safety practices:
        - Maintain updated medication list including OTC drugs and supplements
        - Use same pharmacy when possible for interaction screening
        - Read medication labels and follow dosing instructions exactly
        - Don't share prescription medications
        - Store medications properly (temperature, light, moisture)
        - Check expiration dates regularly
        
        Common interaction examples:
        - Blood thinners with aspirin (increased bleeding risk)
        - Certain antibiotics with dairy products (reduced absorption)
        - Some heart medications with grapefruit (altered drug levels)
        
        Always inform all healthcare providers about every medication and supplement you take.
        """,
        "category": "medication",
        "source": "FDA Drug Safety Guidelines",
        "last_updated": "2024-01-10",
    },
    {
        "id": "emergency_recognition",
        "title": "Medical Emergency Recognition",
        "content": """
        Recognizing medical emergencies can save lives. Call 911 immediately for these symptoms:
        
        Cardiac emergencies:
        - Chest pain or pressure, especially radiating to arm, jaw, or back
        - Sudden severe shortness of breath
        - Rapid or irregular heartbeat with dizziness
        - Fainting or loss of consciousness
        
        Stroke signs (FAST test):
        - Face drooping on one side
        - Arm weakness (can't raise both arms)
        - Speech difficulty or slurred speech
        - Time to call 911 immediately
        
        Respiratory emergencies:
        - Severe difficulty breathing or gasping
        - Choking with inability to speak or cough
        - Blue lips or fingernails (cyanosis)
        
        Other emergencies:
        - Severe bleeding that won't stop
        - Severe allergic reaction (difficulty breathing, swelling)
        - Poisoning or drug overdose
        - Severe burns
        - Head injury with confusion or loss of consciousness
        - Suicidal thoughts or behavior
        
        When in doubt, call emergency services. Don't wait or try to drive yourself.
        """,
        "category": "emergency",
        "source": "American Heart Association",
        "last_updated": "2024-01-12",
    },
    {
        "id": "preventive_healthcare",
        "title": "Preventive Healthcare and Screenings",
        "content": """
        Preventive healthcare helps detect diseases early when they're most treatable.
        
        Regular screenings by age:
        
        Ages 18-39:
        - Annual wellness exam
        - Blood pressure check annually
        - Cholesterol screening every 5 years
        - Diabetes screening if risk factors present
        - Skin cancer checks annually
        - Dental cleanings every 6 months
        
        Ages 40-64:
        - All above screenings
        - Mammograms annually (women)
        - Colonoscopy every 10 years starting at 45
        - Bone density screening (women post-menopause)
        - Prostate screening discussion (men)
        
        Ages 65+:
        - All above screenings
        - Annual flu vaccination
        - Pneumonia vaccination
        - Shingles vaccination
        - More frequent bone density checks
        
        Vaccinations for all ages:
        - COVID-19 boosters as recommended
        - Tdap every 10 years
        - Annual influenza vaccine
        
        Lifestyle prevention:
        - Regular exercise (150 minutes moderate activity weekly)
        - Balanced diet rich in fruits and vegetables
        - Maintain healthy weight
        - Don't smoke or use tobacco
        - Limit alcohol consumption
        - Practice safe sun exposure
        """,
        "category": "prevention",
        "source": "USPSTF Guidelines",
        "last_updated": "2024-01-08",
    },
    {
        "id": "healthcare_navigation",
        "title": "Healthcare System Navigation",
        "content": """
        Understanding healthcare systems helps you get appropriate care efficiently.
        
        Types of care:
        - Primary care: Regular checkups, common illnesses, referrals to specialists
        - Urgent care: Non-emergency issues needing same-day attention
        - Emergency room: Life-threatening conditions requiring immediate care
        - Telehealth: Virtual consultations for non-emergency conditions
        - Specialist care: Focused treatment for specific conditions
        
        Appointment scheduling:
        - Call during business hours for routine appointments
        - Use patient portals for non-urgent requests
        - Ask about same-day sick visits
        - Prepare list of symptoms and questions before appointments
        
        Insurance considerations:
        - Understand your copays, deductibles, and out-of-pocket maximums
        - Check if providers are in-network before scheduling
        - Get prior authorization for procedures when required
        - Keep records of all medical expenses
        
        Patient rights:
        - Right to understand your diagnosis and treatment options
        - Right to participate in treatment decisions
        - Right to access your medical records
        - Right to privacy and confidentiality
        - Right to seek second opinions
        
        Communication tips:
        - Be honest about symptoms and concerns
        - Ask questions if you don't understand
        - Bring medication list to all appointments
        - Follow up on test results
        """,
        "category": "healthcare_navigation",
        "source": "Patient Advocacy Guidelines",
        "last_updated": "2024-01-05",
    },
    {
        "id": "mental_health_resources",
        "title": "Mental Health Resources and Support",
        "content": """
        Mental health is equally important as physical health and requires professional attention.
        
        Common mental health conditions:
        - Depression: Persistent sadness, loss of interest, fatigue, sleep changes
        - Anxiety: Excessive worry, restlessness, physical symptoms like rapid heartbeat
        - PTSD: Following traumatic events, flashbacks, avoidance, hypervigilance
        - Bipolar disorder: Alternating periods of depression and mania
        
        When to seek help:
        - Symptoms interfere with daily activities
        - Thoughts of self-harm or suicide
        - Substance abuse as coping mechanism
        - Relationship or work problems due to mental health
        - Persistent feelings of hopelessness
        
        Treatment options:
        - Therapy (cognitive behavioral, dialectical behavioral, etc.)
        - Medication management with psychiatrist
        - Support groups
        - Lifestyle interventions (exercise, meditation, sleep hygiene)
        
        Crisis resources:
        - National Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741
        - Emergency services: 911
        - Local crisis intervention centers
        
        Reducing stigma:
        - Mental health conditions are medical conditions
        - Treatment is effective for most people
        - Seeking help shows strength, not weakness
        - Recovery is possible with appropriate support
        """,
        "category": "mental_health",
        "source": "National Institute of Mental Health",
        "last_updated": "2024-01-03",
    },
]


# Database Models
@dataclass
class ConversationMessage:
    session_id: str
    message_id: str
    role: str
    content: str
    metadata: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, message_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT NOT NULL,
                    requests_count INTEGER DEFAULT 1,
                    window_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ip_address)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    source TEXT,
                    last_updated DATE,
                    embedding_vector BLOB
                )
            """
            )
            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def save_message(self, message: ConversationMessage):
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversations 
                    (session_id, message_id, role, content, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.session_id,
                        message.message_id,
                        message.role,
                        message.content,
                        message.metadata,
                        message.timestamp,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving message: {e}")

    def get_conversation(self, session_id: str) -> List[ConversationMessage]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT session_id, message_id, role, content, metadata, timestamp
                    FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp ASC
                """,
                    (session_id,),
                )

                return [
                    ConversationMessage(
                        session_id=row[0],
                        message_id=row[1],
                        role=row[2],
                        content=row[3],
                        metadata=row[4],
                        timestamp=datetime.fromisoformat(row[5]) if row[5] else None,
                    )
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}")
            return []


# Vector Store for RAG
class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        self.documents = []
        self.document_metadata = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        try:
            # Extract text content for embedding
            texts = []
            for doc in documents:
                # Combine title and content for better context
                text = f"{doc['title']}\n\n{doc['content']}"
                texts.append(text)
                self.documents.append(text)
                self.document_metadata.append(doc)

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to index
            self.index.add(embeddings)

            logger.info(f"Added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")

    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with improved debugging."""
        try:
            if self.index.ntotal == 0:
                logger.warning("Vector index is empty")
                return []

            logger.info(
                f"Searching for query: '{query[:50]}...' with threshold {threshold}"
            )

            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [query], convert_to_numpy=True
            )
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(
                query_embedding, min(top_k, self.index.ntotal)
            )

            logger.info(f"Search returned scores: {scores[0][:3]}")  # Log top 3 scores

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid index
                    logger.info(
                        f"Checking doc {idx}: score={score:.3f}, threshold={threshold}"
                    )
                    if score >= threshold:
                        results.append(
                            {
                                "document": self.documents[idx],
                                "metadata": self.document_metadata[idx],
                                "score": float(score),
                            }
                        )
                    else:
                        logger.info(
                            f"Document {idx} below threshold ({score:.3f} < {threshold})"
                        )

            logger.info(f"Returning {len(results)} results above threshold")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []


# LLM Interface
class LLMInterface:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        if config.GEMINI_API_KEY:
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)

    def generate_response(
        self, prompt: str, context: str = "", conversation_history: List[Dict] = None
    ) -> str:
        """Generate response using Gemini or fallback."""
        try:
            if self.model and self.config.GEMINI_API_KEY:
                return self._generate_with_gemini(prompt, context, conversation_history)
            else:
                return self._generate_fallback(prompt, context)

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I'm experiencing technical difficulties generating a response. Please try again."

    def _generate_with_gemini(
        self, prompt: str, context: str, conversation_history: List[Dict] = None
    ) -> str:
        """Generate response using Gemini."""
        # Build conversation context
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 4 messages for context
            history_context = "\n".join(
                [
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}..."
                    for msg in recent_history
                ]
            )

        # System prompt for medical chatbot
        system_prompt = """You are MedAssist AI, a professional medical customer service chatbot. Your role is to:

1. Provide accurate, helpful information based on the medical knowledge provided
2. Always include appropriate medical disclaimers
3. Recognize emergency situations and direct users to immediate care
4. Be empathetic and professional
5. Cite sources when possible
6. Never provide specific medical diagnoses or treatment recommendations

IMPORTANT SAFETY RULES:
- Always remind users that this is general information, not personalized medical advice
- For emergency symptoms, immediately direct to emergency services
- Encourage users to consult healthcare providers for specific concerns
- Be clear about the limitations of AI medical assistance

Use the provided medical knowledge context to inform your responses, but present information in a conversational, helpful manner."""

        # Construct full prompt
        full_prompt = f"""
{system_prompt}

MEDICAL KNOWLEDGE CONTEXT:
{context}

RECENT CONVERSATION HISTORY:
{history_context}

CURRENT USER QUESTION:
{prompt}

Please provide a helpful, accurate response based on the medical knowledge provided. Include appropriate disclaimers and safety guidance."""

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback(prompt, context)

    def _generate_fallback(self, prompt: str, context: str) -> str:
        """Fallback response when LLM is not available."""
        emergency_keywords = [
            "emergency",
            "urgent",
            "chest pain",
            "can't breathe",
            "bleeding",
            "unconscious",
        ]

        if any(keyword in prompt.lower() for keyword in emergency_keywords):
            return """🚨 **MEDICAL EMERGENCY DETECTED** 🚨

If this is a life-threatening emergency:
- Call 911 immediately (NIGERIA)
- Go to the nearest emergency room
- Contact emergency services in your country

This system cannot provide emergency medical care. Please seek immediate professional medical attention.

*This is an automated safety response due to LLM service being unavailable.*"""

        if context:
            return f"""Based on the available medical information:

{context[:800]}...

**Important Medical Disclaimer:** This information is for educational purposes only and is not intended as medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for personalized medical guidance.

*Note: Advanced AI response generation is currently unavailable. This is a basic response based on retrieved medical knowledge.*"""
        else:
            return """I don't have specific information about your question in my current knowledge base. 

For the best assistance, I recommend:
- Contacting your healthcare provider directly
- Visiting reputable medical websites like Mayo Clinic or WebMD
- Calling your insurance company's member services for coverage questions

**Important:** This system provides general information only. Always consult healthcare professionals for personalized medical advice.

*Note: Advanced AI response generation is currently unavailable.*"""


# Enhanced RAG System
class MedicalRAGSystem:
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.vector_store = VectorStore(config.EMBEDDING_MODEL)
        self.llm = LLMInterface(config)
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize the vector store with medical knowledge."""
        try:
            logger.info("Initializing medical knowledge base...")
            self.vector_store.add_documents(MEDICAL_KNOWLEDGE_DOCUMENTS)
            logger.info("Medical knowledge base initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")

    def generate_response(
        self, query: str, session_id: str, conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response using RAG pipeline with detailed debugging."""
        try:
            logger.info(f"Processing query: '{query[:50]}...'")

            # Step 1: Emergency Detection
            if self._is_emergency(query):
                logger.info("Emergency detected, returning safety response")
                return {
                    "response": self._get_emergency_response(),
                    "sources": ["emergency_protocol"],
                    "confidence": "critical",
                    "category": "emergency",
                    "method": "emergency_detection",
                }

            # Step 2: Retrieve relevant documents with debugging
            logger.info("Starting vector search...")
            retrieved_docs = self.vector_store.search(
                query,
                top_k=self.config.TOP_K_RETRIEVAL,
                threshold=self.config.SIMILARITY_THRESHOLD,
            )
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # DEBUG: Log retrieval details
            for i, doc in enumerate(retrieved_docs):
                logger.info(
                    f"Doc {i}: {doc['metadata']['title']} (score: {doc['score']:.3f})"
                )

            # Step 3: Check if we have relevant documents
            if not retrieved_docs:
                logger.warning(
                    "No documents retrieved above threshold, lowering threshold"
                )
                # Try with lower threshold
                retrieved_docs = self.vector_store.search(
                    query, top_k=self.config.TOP_K_RETRIEVAL, threshold=0.3
                )
                logger.info(
                    f"Retrieved {len(retrieved_docs)} documents with lower threshold"
                )

            # Step 4: If still no docs, use fallback with knowledge base content
            if not retrieved_docs:
                logger.warning(
                    "No relevant documents found, using general medical knowledge"
                )
                return self._generate_knowledge_fallback(query)

            # Step 5: Prepare context
            context = self._prepare_context(retrieved_docs)
            logger.info(f"Prepared context length: {len(context)} characters")

            # Step 6: Generate response with LLM or fallback
            if self.config.GEMINI_API_KEY and self.llm.model:
                logger.info("Generating response with Gemini")
                response = self.llm.generate_response(
                    prompt=query,
                    context=context,
                    conversation_history=conversation_history,
                )
            else:
                logger.info("Using knowledge-based fallback response")
                response = self._generate_knowledge_based_response(
                    query, retrieved_docs
                )

            # Step 7: Determine confidence and metadata
            confidence = self._calculate_confidence(retrieved_docs, query)
            sources = [doc["metadata"]["id"] for doc in retrieved_docs]
            categories = list(
                set([doc["metadata"]["category"] for doc in retrieved_docs])
            )

            result = {
                "response": response,
                "sources": sources,
                "confidence": confidence,
                "category": categories[0] if categories else "general",
                "method": (
                    "rag_knowledge" if not self.config.GEMINI_API_KEY else "rag_llm"
                ),
                "retrieved_docs_count": len(retrieved_docs),
                "similarity_scores": [doc["score"] for doc in retrieved_docs],
                "context_length": len(context),
            }

            logger.info(
                f"Generated response with method: {result['method']}, confidence: {confidence}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "response": f"I apologize, but I'm experiencing technical difficulties processing your question about medical topics. Please try rephrasing your question or contact support if the issue persists. Error details: {str(e)[:100]}",
                "sources": [],
                "confidence": "error",
                "category": "system_error",
                "method": "error_fallback",
            }

    def _is_emergency(self, query: str) -> bool:
        """Detect emergency situations."""
        emergency_keywords = [
            "emergency",
            "urgent",
            "chest pain",
            "can't breathe",
            "bleeding heavily",
            "unconscious",
            "heart attack",
            "stroke",
            "severe pain",
            "suicide",
            "overdose",
            "choking",
            "severe allergic reaction",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in emergency_keywords)

    def _get_emergency_response(self) -> str:
        """Standard emergency response."""
        return """🚨 **MEDICAL EMERGENCY DETECTED** 🚨

**If this is a life-threatening emergency:**
- **Call 911 immediately** (NIGERIA)
- **Go to the nearest emergency room**
- **Call your local emergency number**

**For urgent but non-life-threatening issues:**
- Contact your healthcare provider
- Visit an urgent care center
- Call a nurse hotline through your insurance

**Crisis Mental Health Support:**
- National Suicide Prevention Lifeline: **988**
- Crisis Text Line: Text **HOME** to **741741**

**⚠️ Important:** This chatbot cannot provide emergency medical care. Always seek immediate professional help for medical emergencies.

Is there anything else I can help you with regarding non-emergency health information?"""

    def _prepare_context(self, retrieved_docs: List[Dict]) -> str:
        """Prepare context from retrieved documents."""
        if not retrieved_docs:
            return ""

        context_parts = []
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            context_parts.append(
                f"**Source: {metadata['title']} ({metadata['source']})**\n"
                f"Category: {metadata['category']}\n"
                f"Content: {doc['document'][:1000]}...\n"
                f"Relevance Score: {doc['score']:.3f}\n"
            )

        return "\n---\n".join(context_parts)

    def _calculate_confidence(self, retrieved_docs: List[Dict], query: str) -> str:
        """Calculate confidence based on retrieval quality."""
        if not retrieved_docs:
            return "low"

        avg_score = sum(doc["score"] for doc in retrieved_docs) / len(retrieved_docs)

        if avg_score >= 0.85:
            return "high"
        elif avg_score >= 0.75:
            return "medium"
        else:
            return "low"

    def _generate_knowledge_fallback(self, query: str) -> Dict[str, Any]:
        """Generate response when no documents are retrieved."""
        query_lower = query.lower()
        relevant_docs = []

        # Simple keyword matching
        keywords = [
            "fever",
            "cough",
            "respiratory",
            "symptom",
            "infection",
            "treatment",
            "emergency",
        ]

        for doc in MEDICAL_KNOWLEDGE_DOCUMENTS:
            doc_content = doc["content"].lower()
            if any(
                keyword in query_lower and keyword in doc_content
                for keyword in keywords
            ):
                relevant_docs.append(doc)

        if relevant_docs:
            doc = relevant_docs[0]
            response = f"""Based on medical information about **{doc['title']}**:

    {doc['content'][:500]}...

    **Source:** {doc['source']}
    **Category:** {doc['category'].replace('_', ' ').title()}

    **⚠️ Important Medical Disclaimer:** This information is for educational purposes only and is not intended as medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for personalized medical guidance.

    **Note:** Response generated using keyword matching due to vector search limitations."""

            return {
                "response": response,
                "sources": [doc["id"]],
                "confidence": "medium",
                "category": doc["category"],
                "method": "keyword_fallback",
                "retrieved_docs_count": 1,
                "similarity_scores": [0.8],
            }

        return {
            "response": """I apologize, but I don't have specific information matching your question in my current medical knowledge base.

    **For fever-related symptoms**, I recommend:
    • Monitor your temperature regularly
    • Stay hydrated with plenty of fluids
    • Rest and avoid strenuous activity
    • Contact your healthcare provider if fever exceeds 103°F (39.4°C)
    • Seek immediate care for difficulty breathing, chest pain, or severe symptoms

    **⚠️ Important:** If this is an emergency, call 911 immediately.

    **General Guidance:**
    • Contact your healthcare provider directly for medical concerns
    • Visit reputable medical websites like Mayo Clinic or WebMD
    • Call your insurance company's member services for coverage questions

    This system provides general information only. Always consult healthcare professionals for personalized medical advice.""",
            "sources": ["general_medical_guidance"],
            "confidence": "low",
            "category": "general",
            "method": "general_fallback",
        }


def _generate_knowledge_based_response(
    self, query: str, retrieved_docs: List[Dict]
) -> str:
    """Generate response based on retrieved documents without LLM."""
    if not retrieved_docs:
        return "I don't have specific relevant information for your question."

    best_doc = retrieved_docs[0]
    metadata = best_doc["metadata"]

    # Extract relevant content
    content = best_doc["document"]
    if len(content) > 800:
        content = content[:800] + "..."

    response = f"""Based on medical information about **{metadata['title']}**:

{content}

**Source:** {metadata['source']}
**Category:** {metadata['category'].replace('_', ' ').title()}

**⚠️ Important Medical Disclaimer:** This information is for educational purposes only and is not intended as medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for personalized medical guidance.

**For immediate medical emergencies, call 911.**"""

    return response


# Rate Limiting (same as before)
class RateLimiter:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def is_rate_limited(self, ip_address: str) -> bool:
        # Implementation same as before
        return False  # Simplified for demo


# Initialize components
@st.cache_resource
def get_components():
    """Initialize and cache application components."""
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        rag_system = MedicalRAGSystem(config, db_manager)
        rate_limiter = RateLimiter(db_manager)
        return db_manager, rag_system, rate_limiter
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        st.error(f"Failed to initialize system components: {e}")
        return None, None, None


# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title=f"{config.APP_NAME} - Medical RAG Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .rag-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .emergency-alert {
        background-color: #ff4444;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border: 2px solid #cc0000;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        f"""
    <div class="main-header">
        <h1>🏥 {config.APP_NAME}</h1>
        <p>Retrieval-Augmented Generation Medical Assistant</p>
        <p><small>Powered by Vector Search + {config.GEMINI_MODEL} • Version {config.VERSION}</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize components
    components = get_components()
    if components[0] is None:  # Failed to initialize
        st.stop()

    db_manager, rag_system, rate_limiter = components

    # Session state
    # Continuation from the truncated code - Session State and UI Implementation

    # Session state initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            f"{datetime.now()}{os.urandom(16)}".encode()
        ).hexdigest()[:16]

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Load conversation history
        history = db_manager.get_conversation(st.session_state.session_id)
        st.session_state.messages = [
            {"role": msg.role, "content": msg.content, "metadata": msg.metadata}
            for msg in history[-config.MAX_CONVERSATION_LENGTH :]
        ]

    if "rag_debug" not in st.session_state:
        st.session_state.rag_debug = False

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ System Controls")

        # System Status
        st.markdown("#### System Status")
        gemini_status = "✅ Connected" if config.GEMINI_API_KEY else "⚠️ Demo Mode"
        st.markdown(f"**LLM Status:** {gemini_status}")
        st.markdown(
            f"**Vector Store:** ✅ {len(rag_system.vector_store.documents)} docs"
        )
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

        st.divider()

        # RAG Configuration
        st.markdown("#### RAG Settings")
        debug_mode = st.toggle("Debug Mode", value=st.session_state.rag_debug)
        st.session_state.rag_debug = debug_mode

        if debug_mode:
            st.markdown("**Debug Info:**")
            st.markdown(f"- Model: {config.EMBEDDING_MODEL}")
            st.markdown(f"- Top-K: {config.TOP_K_RETRIEVAL}")
            st.markdown(f"- Threshold: {config.SIMILARITY_THRESHOLD}")
            st.markdown(f"- Max Tokens: {config.MAX_TOKENS}")

        st.divider()

        # Conversation Management
        st.markdown("#### Conversation")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = hashlib.md5(
                f"{datetime.now()}{os.urandom(16)}".encode()
            ).hexdigest()[:16]
            st.rerun()

        if st.button("📁 Export Chat", use_container_width=True):
            export_conversation(st.session_state.messages, st.session_state.session_id)

        st.divider()

        # Quick Actions
        st.markdown("#### Quick Medical Topics")
        sample_questions = [
            "What are the symptoms of respiratory infections?",
            "How can I prevent heart disease?",
            "When should I seek emergency medical care?",
            "What are common drug interactions?",
            "How do I navigate the healthcare system?",
            "What mental health resources are available?",
        ]

        for i, question in enumerate(sample_questions):
            if st.button(
                f"💬 {question[:25]}...", key=f"sample_{i}", use_container_width=True
            ):
                st.session_state.sample_question = question
                st.rerun()

        st.divider()

        # Medical Disclaimer
        st.markdown("#### ⚠️ Medical Disclaimer")
        st.markdown(
            """
        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;'>
        <small><strong>Important:</strong> This chatbot provides general health information only. 
        Always consult qualified healthcare professionals for personalized medical advice, 
        diagnosis, or treatment decisions.</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Main chat interface
    st.markdown("### 💬 Chat with MedAssist AI")

    # RAG Information Panel
    if st.session_state.rag_debug:
        st.markdown(
            """
        <div class="rag-info">
            <h4>🔍 RAG System Active</h4>
            <p>Responses are enhanced with retrieval-augmented generation using vector search 
            through medical knowledge documents.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # Show debug info for assistant messages
                if (
                    message["role"] == "assistant"
                    and st.session_state.rag_debug
                    and message.get("metadata")
                ):

                    try:
                        metadata = (
                            json.loads(message["metadata"])
                            if isinstance(message["metadata"], str)
                            else message["metadata"]
                        )
                        display_debug_info(metadata)
                    except:
                        pass

    # Handle sample questions
    user_input = ""
    if hasattr(st.session_state, "sample_question"):
        user_input = st.session_state.sample_question
        delattr(st.session_state, "sample_question")

    # Chat input
    if (
        prompt := st.chat_input(
            "Ask me anything about health and medical topics...", key="chat_input"
        )
        or user_input
    ):
        # Check rate limiting
        client_ip = get_client_ip()
        if rate_limiter.is_rate_limited(client_ip):
            st.error("Rate limit exceeded. Please wait before sending another message.")
            return

        # Add user message
        user_message = ConversationMessage(
            session_id=st.session_state.session_id,
            message_id=generate_message_id(),
            role="user",
            content=prompt,
            metadata=json.dumps({"client_ip": client_ip}),
        )

        st.session_state.messages.append(
            {"role": "user", "content": prompt, "metadata": user_message.metadata}
        )
        db_manager.save_message(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(
                "Analyzing your question and searching medical knowledge..."
            ):

                # Get conversation history for context
                conversation_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-6:]  # Last 6 messages
                ]

                # Generate response using RAG
                start_time = time.time()
                rag_response = rag_system.generate_response(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    conversation_history=conversation_history[
                        :-1
                    ],  # Exclude current user message
                )
                response_time = time.time() - start_time

                # Display response
                st.write(rag_response["response"])

                # Show debug information
                if st.session_state.rag_debug:
                    rag_response["response_time"] = response_time
                    display_debug_info(rag_response)

        # Save assistant response
        assistant_message = ConversationMessage(
            session_id=st.session_state.session_id,
            message_id=generate_message_id(),
            role="assistant",
            content=rag_response["response"],
            metadata=json.dumps(
                {
                    "rag_metadata": rag_response,
                    "response_time": response_time,
                    "sources": rag_response.get("sources", []),
                    "confidence": rag_response.get("confidence", "unknown"),
                }
            ),
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": rag_response["response"],
                "metadata": assistant_message.metadata,
            }
        )
        db_manager.save_message(assistant_message)

        st.rerun()

    # Emergency banner
    st.markdown("---")
    st.markdown(
        """
    <div class="emergency-alert">
        <h4>🚨 Medical Emergency?</h4>
        <p><strong>Call 911 immediately</strong> for life-threatening emergencies</p>
        <p>Crisis Support: <strong>988</strong> (Suicide & Crisis Lifeline)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Session Stats**")
        st.markdown(f"Messages: {len(st.session_state.messages)}")
        st.markdown(f"Session: {st.session_state.session_id}")

    with col2:
        st.markdown("**🔧 System Info**")
        st.markdown(f"Version: {config.VERSION}")
        st.markdown(f"Model: {config.GEMINI_MODEL}")

    with col3:
        st.markdown("**📚 Knowledge Base**")
        st.markdown(f"Documents: {len(rag_system.vector_store.documents)}")
        st.markdown(
            f"Categories: {len(set(doc['category'] for doc in MEDICAL_KNOWLEDGE_DOCUMENTS))}"
        )


# Utility Functions
def display_debug_info(metadata: Dict[str, Any]):
    """Display debug information for RAG responses."""
    with st.expander("🔍 Debug Information", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Response Metadata:**")
            st.markdown(f"- **Method:** {metadata.get('method', 'unknown')}")
            st.markdown(f"- **Category:** {metadata.get('category', 'unknown')}")

            confidence = metadata.get("confidence", "unknown")
            confidence_class = f"confidence-{confidence}"
            st.markdown(
                f"- **Confidence:** <span class='{confidence_class}'>{confidence.upper()}</span>",
                unsafe_allow_html=True,
            )

            if "response_time" in metadata:
                st.markdown(f"- **Response Time:** {metadata['response_time']:.2f}s")

            if "retrieved_docs_count" in metadata:
                st.markdown(f"- **Retrieved Docs:** {metadata['retrieved_docs_count']}")

        with col2:
            st.markdown("**Sources Used:**")
            sources = metadata.get("sources", [])
            if sources:
                for source in sources[:5]:  # Show max 5 sources
                    st.markdown(f"- `{source}`")
            else:
                st.markdown("- No specific sources")

            if "similarity_scores" in metadata:
                st.markdown("**Similarity Scores:**")
                scores = metadata["similarity_scores"][:3]  # Top 3 scores
                for i, score in enumerate(scores):
                    st.markdown(f"- Doc {i+1}: {score:.3f}")


def export_conversation(messages: List[Dict], session_id: str):
    """Export conversation to downloadable format."""
    try:
        export_data = {
            "session_id": session_id,
            "export_time": datetime.now().isoformat(),
            "messages": messages,
            "system_info": {
                "app_name": config.APP_NAME,
                "version": config.VERSION,
                "model": config.GEMINI_MODEL,
            },
        }

        json_str = json.dumps(export_data, indent=2, default=str)

        st.download_button(
            label="💾 Download Chat History",
            data=json_str,
            file_name=f"medical_chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        st.error("Failed to export conversation")


def generate_message_id() -> str:
    """Generate unique message ID."""
    return hashlib.md5(f"{datetime.now()}{os.urandom(8)}".encode()).hexdigest()[:12]


def get_client_ip() -> str:
    """Get client IP address for rate limiting."""
    try:
        # Try to get real IP from headers (when behind proxy)
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            forwarded_for = st.context.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()

            real_ip = st.context.headers.get("X-Real-IP")
            if real_ip:
                return real_ip

        # Fallback to session-based identification
        return st.session_state.session_id

    except:
        return "unknown"


# Enhanced Rate Limiter Implementation
class RateLimiter:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT requests_count, window_start 
                    FROM rate_limits 
                    WHERE ip_address = ?
                """,
                    (identifier,),
                )

                result = cursor.fetchone()
                current_time = datetime.now()

                if not result:
                    # First request from this identifier
                    conn.execute(
                        """
                        INSERT INTO rate_limits (ip_address, requests_count, window_start)
                        VALUES (?, 1, ?)
                    """,
                        (identifier, current_time),
                    )
                    conn.commit()
                    return False

                requests_count, window_start = result
                window_start = datetime.fromisoformat(window_start)

                # Check if window has expired
                if (
                    current_time - window_start
                ).total_seconds() > config.RATE_LIMIT_WINDOW:
                    # Reset window
                    conn.execute(
                        """
                        UPDATE rate_limits 
                        SET requests_count = 1, window_start = ?
                        WHERE ip_address = ?
                    """,
                        (current_time, identifier),
                    )
                    conn.commit()
                    return False

                # Check if limit exceeded
                if requests_count >= config.RATE_LIMIT_REQUESTS:
                    return True

                # Increment counter
                conn.execute(
                    """
                    UPDATE rate_limits 
                    SET requests_count = requests_count + 1
                    WHERE ip_address = ?
                """,
                    (identifier,),
                )
                conn.commit()
                return False

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False  # Allow request if rate limit check fails


# Application Health Check
def health_check():
    """Perform system health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    try:
        # Check database
        db_manager, _, _ = get_components()
        with db_manager._get_connection() as conn:
            conn.execute("SELECT 1")
        health_status["checks"]["database"] = "✅ Connected"

        # Check LLM
        if config.GEMINI_API_KEY:
            health_status["checks"]["llm"] = "✅ API Key Present"
        else:
            health_status["checks"]["llm"] = "⚠️ Demo Mode"

        # Check vector store
        health_status["checks"][
            "vector_store"
        ] = f"✅ {len(MEDICAL_KNOWLEDGE_DOCUMENTS)} documents"

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        logger.error(f"Health check failed: {e}")

    return health_status


# Error Handler
def handle_error(error: Exception, context: str = ""):
    """Centralized error handling."""
    error_id = hashlib.md5(f"{datetime.now()}{str(error)}".encode()).hexdigest()[:8]
    logger.error(f"Error {error_id} in {context}: {error}")

    return {
        "error_id": error_id,
        "message": "An error occurred. Please try again or contact support.",
        "context": context,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Run health check on startup
        health = health_check()
        if health["status"] == "unhealthy":
            logger.error(f"System health check failed: {health}")
        else:
            logger.info("System health check passed")

        # Run main application
        main()

    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        st.error("Failed to start the application. Please check the logs.")
        st.exception(e)
