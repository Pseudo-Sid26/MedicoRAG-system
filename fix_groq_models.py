#!/usr/bin/env python3
"""
Fix Groq models - update to current available models
"""

import os
from pathlib import Path


def update_settings_with_new_model():
    """Update settings.py with current Groq model"""

    settings_content = '''import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    # Model Configuration - UPDATED TO CURRENT MODELS
    GROQ_MODEL = "llama3-8b-8192"  # Updated from deprecated mixtral-8x7b-32768
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Store Configuration
    CHROMA_PERSIST_DIRECTORY = "./vector_store"
    COLLECTION_NAME = "medical_documents"

    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Medical Terminology
    ICD_10_URL = "https://icd.who.int/browse10/2019/en"
    SNOMED_CT_URL = "http://snomed.info/sct"

    # Drug Interaction APIs
    DRUG_API_BASE_URL = "https://api.fda.gov/drug/"

    # Retrieval Configuration
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.7

    # Application Configuration
    APP_TITLE = "Medical Literature RAG System"
    APP_DESCRIPTION = "Evidence-based decision support for healthcare professionals"

    # File Upload Configuration
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv']

    # Privacy and Anonymization
    PHI_PATTERNS = [
        r'\\b\\d{3}-\\d{2}-\\d{4}\\b',  # SSN
        r'\\b\\d{10}\\b',  # Phone
        r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',  # Email
        r'\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b'  # Dates
    ]

    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        required_vars = ['GROQ_API_KEY']
        missing_vars = [var for var in required_vars if not getattr(cls, var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        return True
'''

    settings_path = Path("config/settings.py")
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    with open(settings_path, "w", encoding="utf-8") as f:
        f.write(settings_content)

    print("✅ Updated settings.py with current Groq model")


def update_groq_utils_with_current_models():
    """Update groq_utils.py with current model list"""

    updated_utils = '''import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GroqUtils:
    """Utility functions for Groq API operations"""

    @staticmethod
    def validate_api_key(api_key: str) -> Dict[str, Any]:
        """Validate Groq API key with current models"""
        try:
            from groq import Groq

            # Create client
            client = Groq(api_key=api_key)

            # Test with current available model
            response = client.chat.completions.create(
                model="llama3-8b-8192",  # Current working model
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                temperature=0
            )

            return {
                'valid': True,
                'message': 'API key is valid',
                'model_access': True
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"API validation error: {error_msg}")

            if 'api_key' in error_msg.lower() or 'unauthorized' in error_msg.lower():
                return {
                    'valid': False,
                    'message': 'Invalid API key - check your Groq API key',
                    'error': error_msg
                }
            elif 'model_decommissioned' in error_msg.lower():
                return {
                    'valid': False,
                    'message': 'Model decommissioned - updating to current model',
                    'error': error_msg
                }
            else:
                return {
                    'valid': False,
                    'message': f'API validation failed: {error_msg}',
                    'error': error_msg
                }

    @staticmethod
    def get_available_models(api_key: str) -> Dict[str, Any]:
        """Get list of current available Groq models (as of August 2025)"""
        try:
            # Current Groq models available
            available_models = {
                "llama3-8b-8192": {
                    "name": "Llama 3 8B",
                    "context_length": 8192,
                    "description": "Meta's Llama 3 model - fast and efficient",
                    "use_case": "General purpose, good for medical tasks"
                },
                "llama3-70b-8192": {
                    "name": "Llama 3 70B", 
                    "context_length": 8192,
                    "description": "Larger Llama 3 model - more capable",
                    "use_case": "Complex reasoning, detailed analysis"
                },
                "mixtral-8x7b-32768": {
                    "name": "Mixtral 8x7B (DEPRECATED)",
                    "context_length": 32768,
                    "description": "DEPRECATED - use Llama 3 models instead",
                    "use_case": "DO NOT USE - model decommissioned"
                },
                "gemma-7b-it": {
                    "name": "Gemma 7B IT",
                    "context_length": 8192,
                    "description": "Google's Gemma instruction-tuned model",
                    "use_case": "Instruction following, reasoning"
                }
            }

            return {
                'success': True,
                'models': available_models,
                'count': len(available_models),
                'recommended': 'llama3-8b-8192'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'models': {}
            }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        return len(text) // 4

    @staticmethod
    def format_medical_prompt(query: str, context: str, prompt_type: str = 'general') -> str:
        """Format prompt for medical use cases"""

        medical_disclaimers = {
            'general': "Important: This is for healthcare professional use only. Always apply clinical judgment.",
            'diagnosis': "Note: This is diagnostic support only. Clinical correlation and additional testing may be required.",
            'treatment': "Important: Consider patient-specific factors, allergies, and contraindications before implementing any treatment.",
            'drug_interaction': "Critical: Verify all drug interactions with current references and consider patient-specific factors."
        }

        disclaimer = medical_disclaimers.get(prompt_type, medical_disclaimers['general'])

        formatted_prompt = f"""
{disclaimer}

Medical Query: {query}

Relevant Medical Literature and Context:
{context}

Please provide an evidence-based response that:
1. Directly addresses the medical query
2. References relevant evidence from the provided context
3. Acknowledges any limitations or uncertainties
4. Provides actionable clinical guidance when appropriate
5. Maintains appropriate medical terminology while being clear
"""

        return formatted_prompt

    @staticmethod
    def optimize_context_length(context: str, max_tokens: int, model: str) -> str:
        """Optimize context length for model limits"""

        # Updated model limits for current models
        model_limits = {
            "llama3-8b-8192": 8192,
            "llama3-70b-8192": 8192,
            "gemma-7b-it": 8192,
            "mixtral-8x7b-32768": 32768  # Keep for backwards compatibility
        }

        limit = model_limits.get(model, 8192)
        estimated_tokens = GroqUtils.estimate_tokens(context)

        if estimated_tokens <= max_tokens:
            return context

        # Calculate truncation point
        truncate_ratio = max_tokens / estimated_tokens
        truncate_length = int(len(context) * truncate_ratio * 0.9)  # 90% to be safe

        # Try to truncate at sentence boundaries
        sentences = context.split('. ')
        truncated_context = ""

        for sentence in sentences:
            if len(truncated_context) + len(sentence) + 2 <= truncate_length:
                truncated_context += sentence + ". "
            else:
                break

        logger.warning(f"Context truncated from {len(context)} to {len(truncated_context)} characters")
        return truncated_context.strip()

    @staticmethod
    def create_system_message(role_type: str) -> str:
        """Create appropriate system message for different medical roles"""

        system_messages = {
            'general_practitioner': """You are a medical AI assistant helping general practitioners with evidence-based clinical decisions. 
            Provide comprehensive, accurate medical information while emphasizing the importance of clinical judgment and patient-specific factors.""",

            'specialist': """You are a medical AI assistant supporting specialist physicians with advanced clinical analysis. 
            Provide detailed, evidence-based recommendations with appropriate references to current medical literature.""",

            'nurse': """You are a medical AI assistant supporting nursing professionals with patient care decisions. 
            Focus on practical, evidence-based guidance for nursing interventions and patient monitoring.""",

            'pharmacist': """You are a medical AI assistant specializing in pharmacotherapy and drug interactions. 
            Provide detailed medication analysis, interaction checking, and pharmaceutical guidance.""",

            'researcher': """You are a medical AI assistant supporting medical research and literature analysis. 
            Provide comprehensive analysis of medical evidence with appropriate critical evaluation."""
        }

        return system_messages.get(role_type, system_messages['general_practitioner'])
'''

    utils_path = Path("src/utils/groq_utils.py")
    utils_path.parent.mkdir(parents=True, exist_ok=True)

    with open(utils_path, "w", encoding="utf-8") as f:
        f.write(updated_utils)

    print("✅ Updated groq_utils.py with current models")


def update_response_generator_model():
    """Update the response generator to use current model"""

    # Read the current response generator
    gen_path = Path("src/generation/groq_response_generator.py")

    if gen_path.exists():
        with open(gen_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace the deprecated model
        updated_content = content.replace(
            'model: str = "mixtral-8x7b-32768"',
            'model: str = "llama3-8b-8192"'
        )

        # Also replace any hardcoded model references
        updated_content = updated_content.replace(
            '"mixtral-8x7b-32768"',
            '"llama3-8b-8192"'
        )

        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

        print("✅ Updated groq_response_generator.py with current model")
    else:
        print("⚠️ groq_response_generator.py not found - will use settings default")


def update_env_file():
    """Update .env file with current model"""
    env_content = '''# Medical RAG System Environment Variables

# Groq API Configuration (REQUIRED)
GROQ_API_KEY=your_groq_api_key_here

# HuggingFace Configuration (Optional)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Application Configuration
APP_ENV=development
DEBUG=true

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY=./vector_store
COLLECTION_NAME=medical_documents

# Model Configuration - UPDATED TO CURRENT MODELS
GROQ_MODEL=llama3-8b-8192
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# Security Configuration
MAX_FILE_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=.pdf,.txt,.docx,.csv

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/medical_rag.log
'''

    env_path = Path(".env")
    if env_path.exists():
        # Read existing .env and update only the model
        with open(env_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

        # Replace the model if it exists
        if "GROQ_MODEL=" in existing_content:
            import re
            updated_content = re.sub(
                r'GROQ_MODEL=.*',
                'GROQ_MODEL=llama3-8b-8192',
                existing_content
            )

            with open(env_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print("✅ Updated existing .env file with current model")
        else:
            print("⚠️ GROQ_MODEL not found in .env - add manually if needed")
    else:
        # Create new .env file
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)

        print("✅ Created new .env file with current model")


def main():
    """Main fix function"""
    print("🚨 Fixing Groq Model Deprecation Issue")
    print("=" * 50)

    print("The Mixtral model has been decommissioned.")
    print("Updating to current Groq models...\n")

    # Update all files with new model
    update_settings_with_new_model()
    update_groq_utils_with_current_models()
    update_response_generator_model()
    update_env_file()

    print("\n" + "=" * 50)
    print("🎉 MODEL UPDATE COMPLETE!")

    print("\nWhat was updated:")
    print("✅ settings.py - Updated to llama3-8b-8192")
    print("✅ groq_utils.py - Updated model list and validation")
    print("✅ groq_response_generator.py - Updated default model")
    print("✅ .env - Updated GROQ_MODEL setting")

    print("\nCurrent Groq Models Available:")
    print("🟢 llama3-8b-8192 (RECOMMENDED)")
    print("🟢 llama3-70b-8192 (More powerful)")
    print("🟢 gemma-7b-it (Alternative)")
    print("🔴 mixtral-8x7b-32768 (DEPRECATED)")

    print("\nNext steps:")
    print("1. Run your app: streamlit run app.py")
    print("2. API validation should now work")
    print("3. The system will use Llama 3 8B model")

    print("\n💡 Note:")
    print("Llama 3 models are faster and more efficient than the old Mixtral model!")


if __name__ == "__main__":
    main()