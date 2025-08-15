
import os
import logging
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Optimized settings configuration for Medical RAG System"""

    # Core API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    # Model Configuration
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector Store
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectordb_store")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_docs")

    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    # Response Configuration
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "800"))
    RESPONSE_TEMPERATURE = float(os.getenv("RESPONSE_TEMPERATURE", "0.1"))

    # File Upload
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv']

    # Application
    APP_TITLE = "Medical Literature RAG System"
    APP_DESCRIPTION = "Evidence-based decision support for healthcare professionals"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Privacy - Essential PHI patterns only
    PHI_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{1,2}/\d{1,2}/\d{4}\b'  # Dates
    ]

    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration only"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Set it in your .env file.")

        # Create vector store directory if needed
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

        # Basic validation
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        return True

    @classmethod
    def setup_logging(cls):
        """Simple logging setup"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    @classmethod
    def get_model_config(cls, model_type: str = "default") -> Dict:
        """Get model configuration"""
        config = {
            "model": cls.GROQ_MODEL,
            "max_tokens": cls.MAX_RESPONSE_TOKENS,
            "temperature": cls.RESPONSE_TEMPERATURE
        }

        # Simplified model variants
        if model_type == "fast":
            config.update({"max_tokens": 600, "temperature": 0.05})
        elif model_type == "detailed":
            config.update({"max_tokens": 1000, "temperature": 0.15})

        return config


# Essential constants only
EVIDENCE_WEIGHTS = {
    'systematic_review': 1.0,
    'rct': 0.8,
    'cohort_study': 0.6,
    'case_control': 0.4,
    'expert_opinion': 0.2
}

CLINICAL_GUIDELINES = [
    "AHA/ACC", "ADA", "NICE", "WHO", "CDC"
]

# Initialize
Settings.setup_logging()