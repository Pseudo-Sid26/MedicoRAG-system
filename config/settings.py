# # import os
# # from dotenv import load_dotenv
# #
# # load_dotenv()
# #
# # class Settings:
# #     # API Keys
# #     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# #     HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# #
# #     # Model Configuration - UPDATED TO CURRENT MODELS
# #     GROQ_MODEL = "llama3-8b-8192"  # Updated from deprecated mixtral-8x7b-32768
# #     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# #
# #     # Vector Store Configuration
# #     CHROMA_PERSIST_DIRECTORY = "./vector_store"
# #     COLLECTION_NAME = "medical_documents"
# #
# #     # Chunking Configuration
# #     CHUNK_SIZE = 1000
# #     CHUNK_OVERLAP = 200
# #
# #     # Medical Terminology
# #     ICD_10_URL = "https://icd.who.int/browse10/2019/en"
# #     SNOMED_CT_URL = "http://snomed.info/sct"
# #
# #     # Drug Interaction APIs
# #     DRUG_API_BASE_URL = "https://api.fda.gov/drug/"
# #
# #     # Retrieval Configuration
# #     TOP_K_RETRIEVAL = 5
# #     SIMILARITY_THRESHOLD = 0.7
# #
# #     # Application Configuration
# #     APP_TITLE = "Medical Literature RAG System"
# #     APP_DESCRIPTION = "Evidence-based decision support for healthcare professionals"
# #
# #     # File Upload Configuration
# #     MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
# #     ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv']
# #
# #     # Privacy and Anonymization
# #     PHI_PATTERNS = [
# #         r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
# #         r'\b\d{10}\b',  # Phone
# #         r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
# #         r'\b\d{1,2}/\d{1,2}/\d{4}\b'  # Dates
# #     ]
# #
# #     @classmethod
# #     def validate_config(cls):
# #         """Validate required configuration"""
# #         required_vars = ['GROQ_API_KEY']
# #         missing_vars = [var for var in required_vars if not getattr(cls, var)]
# #
# #         if missing_vars:
# #             raise ValueError(f"Missing required environment variables: {missing_vars}")
# #
# #         return True
# #
# # # Clinical Decision Support Configuration
# # EVIDENCE_HIERARCHY_WEIGHTS = {
# #     'systematic_review': 1.0,
# #     'rct': 0.8,
# #     'cohort_study': 0.6,
# #     'case_control': 0.4,
# #     'case_series': 0.2,
# #     'expert_opinion': 0.1
# # }
# #
# # # Regulatory Compliance
# # FDA_API_ENDPOINT = "https://api.fda.gov/drug/label.json"
# # REGULATORY_CHECK_ENABLED = True
# #
# # # PHI Detection Sensitivity
# # PHI_DETECTION_SENSITIVITY = "high"  # high, medium, low
# # ANONYMIZATION_METHOD = "hash"  # hash, replace, remove
# #
# # # Clinical Guidelines Sources
# # CLINICAL_GUIDELINES_SOURCES = [
# #     "AHA/ACC",
# #     "ADA",
# #     "NICE",
# #     "WHO",
# #     "CDC"
# # ]
#
# import os
# import logging
# from typing import Dict, List, Optional
# from dotenv import load_dotenv
#
# load_dotenv()
#
#
# class Settings:
#     # API Keys
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
#
#     # Model Configuration - UPDATED TO CURRENT MODELS
#     GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Allow env override
#     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
#
#     # Alternative models for failover
#     GROQ_MODELS = {
#         "fast": "llama3-8b-8192",
#         "accurate": "llama3-70b-8192",
#         "alternative": "gemma-7b-it"
#     }
#
#     # Vector Store Configuration
#     CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store")
#     COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_documents")
#
#     # Performance: Vector store settings
#     CHROMA_SETTINGS = {
#         "chroma_db_impl": "duckdb+parquet",  # Better performance than SQLite
#         "anonymized_telemetry": False,
#         "allow_reset": True
#     }
#
#     # HNSW Index settings for better search performance
#     HNSW_CONFIG = {
#         "hnsw:space": "cosine",
#         "hnsw:construction_ef": int(os.getenv("HNSW_CONSTRUCTION_EF", "200")),
#         "hnsw:M": int(os.getenv("HNSW_M", "16")),
#         "hnsw:max_elements": int(os.getenv("HNSW_MAX_ELEMENTS", "50000"))
#     }
#
#     # Chunking Configuration
#     CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
#     CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
#
#     # Performance: Batch processing settings
#     EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
#     PROCESSING_MAX_WORKERS = int(os.getenv("PROCESSING_MAX_WORKERS", "3"))
#
#     # Medical Terminology
#     ICD_10_URL = "https://icd.who.int/browse10/2019/en"
#     SNOMED_CT_URL = "http://snomed.info/sct"
#
#     # Drug Interaction APIs
#     DRUG_API_BASE_URL = "https://api.fda.gov/drug/"
#
#     # Retrieval Configuration
#     TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
#     SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
#
#     # Performance: Search optimization
#     SEARCH_CACHE_SIZE = int(os.getenv("SEARCH_CACHE_SIZE", "100"))
#     SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))  # 1 hour
#
#     # Application Configuration
#     APP_TITLE = "Medical Literature RAG System"
#     APP_DESCRIPTION = "Evidence-based decision support for healthcare professionals"
#
#     # File Upload Configuration
#     MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
#     ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv']
#
#     # Performance: Concurrent file processing
#     MAX_CONCURRENT_UPLOADS = int(os.getenv("MAX_CONCURRENT_UPLOADS", "5"))
#
#     # Privacy and Anonymization
#     PHI_PATTERNS = [
#         r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
#         r'\b\d{10}\b',  # Phone
#         r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
#         r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates
#         r'\b\d{4}-\d{2}-\d{2}\b',  # ISO dates
#         r'\b[A-Z]{2}\d{6}\b'  # Medical record numbers
#     ]
#
#     # Performance: Caching Configuration
#     ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
#     QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "50"))
#     QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "1800"))  # 30 minutes
#
#     ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
#     EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "500"))
#
#     # Performance: Response Generation
#     MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "800"))
#     RESPONSE_TEMPERATURE = float(os.getenv("RESPONSE_TEMPERATURE", "0.1"))
#     RESPONSE_TOP_P = float(os.getenv("RESPONSE_TOP_P", "0.9"))
#
#     # Logging Configuration
#     LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
#     ENABLE_DEBUG = os.getenv("ENABLE_DEBUG", "false").lower() == "true"
#     LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#
#     # Memory Management
#     ENABLE_MEMORY_MONITORING = os.getenv("ENABLE_MEMORY_MONITORING", "true").lower() == "true"
#     MEMORY_CLEANUP_THRESHOLD = int(os.getenv("MEMORY_CLEANUP_THRESHOLD", "80"))  # 80% memory usage
#     PERIODIC_CLEANUP_INTERVAL = int(os.getenv("PERIODIC_CLEANUP_INTERVAL", "25"))  # Every 25 page loads
#
#     # Session Management
#     SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour
#     MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "15"))
#
#     # API Rate Limiting
#     GROQ_RATE_LIMIT = int(os.getenv("GROQ_RATE_LIMIT", "50"))  # Requests per minute
#     API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
#     API_RETRY_DELAY = float(os.getenv("API_RETRY_DELAY", "1.0"))
#
#     # Streamlit Optimization
#     STREAMLIT_CONFIG = {
#         "server.maxUploadSize": 200,
#         "server.enableCORS": False,
#         "server.enableXsrfProtection": False,
#         "browser.gatherUsageStats": False,
#         "runner.fastReruns": True,
#         "runner.magicEnabled": False,
#         "client.caching": True,
#         "client.showErrorDetails": False,
#         "global.developmentMode": False,
#         "global.suppressDeprecationWarnings": True,
#         "logger.level": "ERROR"
#     }
#
#     @classmethod
#     def validate_config(cls):
#         """Validate required configuration with detailed error messages"""
#         errors = []
#
#         # Check required API keys
#         if not cls.GROQ_API_KEY:
#             errors.append("GROQ_API_KEY is required. Please set it in your .env file or environment variables.")
#
#         # Validate directory paths
#         if not os.path.exists(os.path.dirname(cls.CHROMA_PERSIST_DIRECTORY)):
#             try:
#                 os.makedirs(os.path.dirname(cls.CHROMA_PERSIST_DIRECTORY), exist_ok=True)
#             except Exception as e:
#                 errors.append(f"Cannot create vector store directory: {e}")
#
#         # Validate numeric ranges
#         if cls.CHUNK_SIZE < 100 or cls.CHUNK_SIZE > 5000:
#             errors.append("CHUNK_SIZE must be between 100 and 5000")
#
#         if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
#             errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
#
#         if cls.TOP_K_RETRIEVAL < 1 or cls.TOP_K_RETRIEVAL > 20:
#             errors.append("TOP_K_RETRIEVAL must be between 1 and 20")
#
#         if cls.SIMILARITY_THRESHOLD < 0.0 or cls.SIMILARITY_THRESHOLD > 1.0:
#             errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
#
#         # Validate model availability
#         if cls.GROQ_MODEL not in cls.GROQ_MODELS.values():
#             logging.warning(f"GROQ_MODEL '{cls.GROQ_MODEL}' not in recommended models list")
#
#         if errors:
#             raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
#
#         return True
#
#     @classmethod
#     def get_cache_config(cls) -> Dict:
#         """Get caching configuration"""
#         return {
#             "query_cache": {
#                 "enabled": cls.ENABLE_QUERY_CACHE,
#                 "size": cls.QUERY_CACHE_SIZE,
#                 "ttl": cls.QUERY_CACHE_TTL
#             },
#             "embedding_cache": {
#                 "enabled": cls.ENABLE_EMBEDDING_CACHE,
#                 "size": cls.EMBEDDING_CACHE_SIZE
#             },
#             "search_cache": {
#                 "size": cls.SEARCH_CACHE_SIZE,
#                 "ttl": cls.SEARCH_CACHE_TTL
#             }
#         }
#
#     @classmethod
#     def get_performance_config(cls) -> Dict:
#         """Get performance optimization configuration"""
#         return {
#             "embedding_batch_size": cls.EMBEDDING_BATCH_SIZE,
#             "max_workers": cls.PROCESSING_MAX_WORKERS,
#             "max_concurrent_uploads": cls.MAX_CONCURRENT_UPLOADS,
#             "memory_monitoring": cls.ENABLE_MEMORY_MONITORING,
#             "cleanup_threshold": cls.MEMORY_CLEANUP_THRESHOLD,
#             "periodic_cleanup": cls.PERIODIC_CLEANUP_INTERVAL
#         }
#
#     @classmethod
#     def setup_logging(cls):
#         """Setup logging configuration"""
#         logging.basicConfig(
#             level=getattr(logging, cls.LOG_LEVEL.upper()),
#             format=cls.LOG_FORMAT,
#             handlers=[
#                 logging.StreamHandler(),
#                 logging.FileHandler("medical_rag.log") if cls.ENABLE_DEBUG else logging.NullHandler()
#             ]
#         )
#
#     @classmethod
#     def get_model_config(cls, model_type: str = "default") -> Dict:
#         """Get model configuration for different use cases"""
#         base_config = {
#             "model": cls.GROQ_MODEL,
#             "max_tokens": cls.MAX_RESPONSE_TOKENS,
#             "temperature": cls.RESPONSE_TEMPERATURE,
#             "top_p": cls.RESPONSE_TOP_P
#         }
#
#         if model_type == "fast":
#             base_config.update({
#                 "model": cls.GROQ_MODELS["fast"],
#                 "max_tokens": 600,
#                 "temperature": 0.05
#             })
#         elif model_type == "accurate":
#             base_config.update({
#                 "model": cls.GROQ_MODELS["accurate"],
#                 "max_tokens": 1000,
#                 "temperature": 0.1
#             })
#
#         return base_config
#
#
# # Clinical Decision Support Configuration (Enhanced)
# EVIDENCE_HIERARCHY_WEIGHTS = {
#     'systematic_review': 1.0,
#     'meta_analysis': 0.95,
#     'rct': 0.8,
#     'cohort_study': 0.6,
#     'case_control': 0.4,
#     'case_series': 0.2,
#     'case_report': 0.15,
#     'expert_opinion': 0.1,
#     'clinical_guideline': 0.85,
#     'consensus_statement': 0.3
# }
#
# # Regulatory Compliance (Enhanced)
# FDA_API_ENDPOINT = "https://api.fda.gov/drug/label.json"
# REGULATORY_CHECK_ENABLED = os.getenv("REGULATORY_CHECK_ENABLED", "true").lower() == "true"
# REGULATORY_CACHE_TTL = int(os.getenv("REGULATORY_CACHE_TTL", "86400"))  # 24 hours
#
# # PHI Detection Sensitivity (Enhanced)
# PHI_DETECTION_SENSITIVITY = os.getenv("PHI_DETECTION_SENSITIVITY", "high")  # high, medium, low
# ANONYMIZATION_METHOD = os.getenv("ANONYMIZATION_METHOD", "hash")  # hash, replace, remove
# ENABLE_PHI_DETECTION = os.getenv("ENABLE_PHI_DETECTION", "true").lower() == "true"
#
# # Clinical Guidelines Sources (Enhanced)
# CLINICAL_GUIDELINES_SOURCES = [
#     "AHA/ACC",
#     "ADA",
#     "NICE",
#     "WHO",
#     "CDC",
#     "USPSTF",
#     "ACOG",
#     "AAP",
#     "NCCN",
#     "ESC/ERS"
# ]
#
# # Performance Monitoring
# ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"
# PERFORMANCE_LOG_INTERVAL = int(os.getenv("PERFORMANCE_LOG_INTERVAL", "100"))  # Log every N operations
#
# # Quality Assurance
# ENABLE_RESPONSE_VALIDATION = os.getenv("ENABLE_RESPONSE_VALIDATION", "true").lower() == "true"
# MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6"))
# ENABLE_FACT_CHECKING = os.getenv("ENABLE_FACT_CHECKING", "false").lower() == "true"
#
# # Initialize logging when module is imported
# Settings.setup_logging()

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
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store")
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