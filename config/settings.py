import os
import logging
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

def get_secret(key: str, default=None):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        # Try to get from Streamlit secrets first (for cloud deployment)
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    # Fallback to environment variables (for local development)
    return os.getenv(key, default)


class Settings:
    """Optimized settings configuration for Medical RAG System with FAISS"""

    # Core API Configuration
    GROQ_API_KEY = get_secret("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = get_secret("HUGGINGFACE_API_KEY")

    # Model Configuration
    GROQ_MODEL = get_secret("GROQ_MODEL", "llama3-8b-8192")
    EMBEDDING_MODEL = get_secret("EMBEDDING_MODEL", "tfidf")  # Default to TF-IDF for stability

    # Vector Store Configuration (FAISS)
    VECTOR_STORE_DIRECTORY = get_secret("VECTOR_STORE_DIRECTORY", "./vectordb_store")
    CHROMA_PERSIST_DIRECTORY = get_secret("VECTOR_STORE_DIRECTORY", "./vectordb_store")  # For compatibility
    COLLECTION_NAME = get_secret("COLLECTION_NAME", "medical_documents")
    EMBEDDING_DIMENSION = int(get_secret("EMBEDDING_DIMENSION", "384"))

    # Processing Configuration
    CHUNK_SIZE = int(get_secret("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(get_secret("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(get_secret("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD = float(get_secret("SIMILARITY_THRESHOLD", "0.3"))  # Lowered for medical content

    # Response Configuration
    MAX_RESPONSE_TOKENS = int(get_secret("MAX_RESPONSE_TOKENS", "800"))
    RESPONSE_TEMPERATURE = float(get_secret("RESPONSE_TEMPERATURE", "0.1"))

    # File Upload Configuration
    MAX_FILE_SIZE = int(get_secret("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv', '.md']

    # Application Configuration
    APP_TITLE = "Medical Literature RAG System"
    APP_DESCRIPTION = "Evidence-based decision support for healthcare professionals"
    APP_VERSION = "2.0.0"

    # Logging Configuration
    LOG_LEVEL = get_secret("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Medical Content Configuration
    MEDICAL_SPECIALTIES = [
        "cardiology", "endocrinology", "neurology", "oncology",
        "pulmonology", "gastroenterology", "nephrology", "rheumatology",
        "infectious_disease", "emergency_medicine", "family_medicine",
        "internal_medicine", "pediatrics", "obstetrics_gynecology",
        "ophthalmology", "dermatology", "psychiatry", "surgery"
    ]

    DOCUMENT_TYPES = [
        "clinical_guideline", "research_paper", "case_study",
        "systematic_review", "meta_analysis", "treatment_protocol",
        "diagnostic_criteria", "drug_information", "patient_education"
    ]

    # Privacy - Essential PHI patterns only
    PHI_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone numbers
        r'\bMRN\s*:?\s*\d+\b',  # Medical record numbers
    ]

    # Search Configuration
    SEARCH_CONTEXTS = {
        "diagnosis": {
            "keywords": ["diagnosis", "diagnostic", "symptoms", "signs", "criteria"],
            "weight": 1.2
        },
        "treatment": {
            "keywords": ["treatment", "therapy", "medication", "intervention", "management"],
            "weight": 1.1
        },
        "general": {
            "keywords": ["medical", "clinical", "patient", "healthcare"],
            "weight": 1.0
        }
    }

    # Response Templates
    RESPONSE_TEMPLATES = {
        "medical_analysis": """
        **🔬 Medical Analysis**

        **📖 Detailed Analysis**
        {analysis}

        **💊 Clinical Recommendations**
        {recommendations}

        **⚠️ Limitations & Considerations**
        {limitations}
        """,

        "error_response": """
        **🔬 Medical Analysis**

        **📖 Detailed Analysis**
        I apologize, but I cannot provide a comprehensive response based on the available information. 

        **💊 Clinical Recommendations**
        Please consult with a healthcare professional for specific medical guidance.

        **⚠️ Limitations & Considerations**
        This system is designed for healthcare professionals and should not replace clinical judgment.
        """,

        "no_documents": """
        **🔬 Medical Analysis**

        **📖 Detailed Analysis**
        Based on general medical knowledge: {general_response}

        **💊 Clinical Recommendations**
        {recommendations}

        **⚠️ Limitations & Considerations**
        This response is based on general medical knowledge. Please verify with current clinical guidelines and consult appropriate specialists.
        """
    }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration"""
        try:
            # Check required API key
            if not cls.GROQ_API_KEY:
                error_msg = (
                    "GROQ_API_KEY is required. "
                    "For local development: Set it in your .env file. "
                    "For Streamlit Cloud: Add it to your app's secrets in the Streamlit Cloud dashboard."
                )
                raise ValueError(error_msg)

            # Create necessary directories
            os.makedirs(cls.VECTOR_STORE_DIRECTORY, exist_ok=True)

            # Validate chunk configuration
            if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
                raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

            if cls.SIMILARITY_THRESHOLD < 0 or cls.SIMILARITY_THRESHOLD > 1:
                raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")

            # Validate file size limits
            if cls.MAX_FILE_SIZE <= 0:
                raise ValueError("MAX_FILE_SIZE must be positive")

            logging.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            raise

    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)

            # Configure logging
            logging.basicConfig(
                level=getattr(logging, cls.LOG_LEVEL.upper()),
                format=cls.LOG_FORMAT,
                handlers=[
                    logging.StreamHandler(),  # Console output
                    logging.FileHandler(
                        os.path.join(log_dir, "medical_rag.log"),
                        mode='a',
                        encoding='utf-8'
                    )
                ]
            )

            # Set specific logger levels
            logging.getLogger("faiss").setLevel(logging.WARNING)
            logging.getLogger("sklearn").setLevel(logging.WARNING)
            logging.getLogger("numpy").setLevel(logging.WARNING)

            logging.info("✅ Logging configuration initialized")

        except Exception as e:
            print(f"❌ Failed to setup logging: {e}")
            # Fallback to basic config
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )

    @classmethod
    def get_model_config(cls, model_type: str = "default") -> Dict:
        """Get model configuration for different use cases"""
        base_config = {
            "model": cls.GROQ_MODEL,
            "max_tokens": cls.MAX_RESPONSE_TOKENS,
            "temperature": cls.RESPONSE_TEMPERATURE
        }

        # Model configurations for different scenarios
        configs = {
            "default": base_config,
            "fast": {
                **base_config,
                "max_tokens": 600,
                "temperature": 0.05
            },
            "detailed": {
                **base_config,
                "max_tokens": 1000,
                "temperature": 0.15
            },
            "conservative": {
                **base_config,
                "max_tokens": 800,
                "temperature": 0.0
            },
            "exploratory": {
                **base_config,
                "max_tokens": 1200,
                "temperature": 0.2
            }
        }

        return configs.get(model_type, base_config)

    @classmethod
    def get_embedding_config(cls) -> Dict:
        """Get embedding configuration"""
        return {
            "model_name": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "cache_dir": os.path.join(cls.VECTOR_STORE_DIRECTORY, "models")
        }

    @classmethod
    def get_search_config(cls, context: str = "general") -> Dict:
        """Get search configuration for different medical contexts"""
        base_config = {
            "top_k": cls.TOP_K_RETRIEVAL,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "collection_name": cls.COLLECTION_NAME
        }

        # Context-specific adjustments
        if context in cls.SEARCH_CONTEXTS:
            context_config = cls.SEARCH_CONTEXTS[context]
            base_config.update({
                "context_keywords": context_config["keywords"],
                "relevance_weight": context_config["weight"]
            })

        return base_config

    @classmethod
    def get_file_config(cls) -> Dict:
        """Get file processing configuration"""
        return {
            "max_size": cls.MAX_FILE_SIZE,
            "allowed_extensions": cls.ALLOWED_EXTENSIONS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }

    @classmethod
    def is_medical_specialty_valid(cls, specialty: str) -> bool:
        """Check if medical specialty is valid"""
        return specialty.lower() in [s.lower() for s in cls.MEDICAL_SPECIALTIES]

    @classmethod
    def is_document_type_valid(cls, doc_type: str) -> bool:
        """Check if document type is valid"""
        return doc_type.lower() in [d.lower() for d in cls.DOCUMENT_TYPES]

    @classmethod
    def get_response_template(cls, template_type: str = "medical_analysis") -> str:
        """Get response template for formatting"""
        return cls.RESPONSE_TEMPLATES.get(template_type, cls.RESPONSE_TEMPLATES["error_response"])

    @classmethod
    def get_system_info(cls) -> Dict:
        """Get system configuration information"""
        return {
            "app_title": cls.APP_TITLE,
            "app_description": cls.APP_DESCRIPTION,
            "version": cls.APP_VERSION,
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "vector_store": "FAISS",
            "llm_model": cls.GROQ_MODEL,
            "supported_specialties": len(cls.MEDICAL_SPECIALTIES),
            "supported_document_types": len(cls.DOCUMENT_TYPES)
        }


# Enhanced evidence weights for medical literature
EVIDENCE_WEIGHTS = {
    'systematic_review': 1.0,
    'meta_analysis': 0.95,
    'rct': 0.8,
    'cohort_study': 0.6,
    'case_control': 0.4,
    'case_series': 0.3,
    'expert_opinion': 0.2,
    'clinical_guideline': 0.9,
    'treatment_protocol': 0.85
}

# Recognized clinical guidelines and organizations
CLINICAL_GUIDELINES = [
    "AHA/ACC", "ADA", "NICE", "WHO", "CDC", "FDA", "EMA",
    "NCCN", "ASCO", "ESC", "ATS", "IDSA", "ACOG", "AAP",
    "ACR", "AACE", "EASD", "ESH", "CCS"
]

# Medical terminology mappings for better search
MEDICAL_SYNONYMS = {
    "myocardial_infarction": ["heart attack", "MI", "STEMI", "NSTEMI"],
    "diabetes_mellitus": ["diabetes", "DM", "T1DM", "T2DM"],
    "hypertension": ["high blood pressure", "HTN", "elevated BP"],
    "hyperlipidemia": ["high cholesterol", "dyslipidemia"],
    "chronic_kidney_disease": ["CKD", "renal failure", "kidney disease"]
}

# Quality indicators for medical documents
QUALITY_INDICATORS = {
    "peer_reviewed": 1.2,
    "recent_publication": 1.1,  # Within last 5 years
    "high_impact_journal": 1.15,
    "multiple_institutions": 1.05,
    "large_sample_size": 1.1
}

# Initialize configuration
try:
    Settings.setup_logging()
    Settings.validate_config()
    logging.info(f"🏥 {Settings.APP_TITLE} v{Settings.APP_VERSION} initialized")
    logging.info(f"📊 Vector store: FAISS | Embedding: {Settings.EMBEDDING_MODEL}")
except Exception as e:
    logging.error(f"❌ Failed to initialize settings: {e}")
    raise