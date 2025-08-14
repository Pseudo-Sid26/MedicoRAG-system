import pytest
import sys
import os

sys.path.append('src')

from src.generation.groq_response_generator import GroqMedicalResponseGenerator
from src.utils.groq_utils import GroqModelManager, GroqResponseOptimizer
from config.settings import settings


class TestGroqIntegration:
    """Test suite for Groq integration"""

    @pytest.fixture
    def groq_generator(self):
        """Fixture for Groq response generator"""
        if not settings.GROQ_API_KEY:
            pytest.skip("GROQ_API_KEY not available")
        return GroqMedicalResponseGenerator()

    @pytest.fixture
    def model_manager(self):
        """Fixture for Groq model manager"""
        if not settings.GROQ_API_KEY:
            pytest.skip("GROQ_API_KEY not available")
        return GroqModelManager()

    def test_groq_connection(self, groq_generator):
        """Test Groq API connection"""
        result = groq_generator.test_connection()
        assert result['status'] == 'success'
        assert 'test_response' in result

    def test_model_switching(self, groq_generator):
        """Test switching between Groq models"""
        original_model = groq_generator.model_name

        # Switch to a different model
        target_model = "llama3-8b-8192"
        success = groq_generator.switch_model(target_model)

        assert success == True
        assert groq_generator.model_name == target_model

        # Switch back
        groq_generator.switch_model(original_model)
        assert groq_generator.model_name == original_model

    def test_medical_response_generation(self, groq_generator):
        """Test medical response generation"""

        # Mock retrieved documents
        retrieved_docs = [
            {
                'content': 'Metformin is used to treat type 2 diabetes.',
                'metadata': {
                    'source': 'medical_literature.pdf',
                    'document_type': 'medical_literature'
                },
                'similarity_score': 0.95
            }
        ]

        result = groq_generator.generate_  # Medical RAG System with Groq API - Complete Implementation