
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GroqUtils:
    """Optimized utility functions for Groq API operations"""

    # Current working models (as of August 2025)
    MODELS = {
        "llama3-8b-8192": {"name": "Llama 3 8B", "context": 8192, "recommended": True},
        "llama3-70b-8192": {"name": "Llama 3 70B", "context": 8192, "recommended": False},
        "gemma-7b-it": {"name": "Gemma 7B IT", "context": 8192, "recommended": False}
    }

    # System prompts for different medical contexts
    SYSTEM_PROMPTS = {
        'general': "You are a medical AI assistant for healthcare professionals. Provide accurate, evidence-based information. Always emphasize clinical judgment.",
        'diagnosis': "You are a medical AI for diagnostic support. Provide differential diagnoses with evidence. Emphasize need for clinical correlation.",
        'treatment': "You are a medical AI for treatment guidance. Consider contraindications, interactions, and patient factors.",
        'drug_interaction': "You are a medical AI for pharmacotherapy. Focus on drug interactions, dosing, and safety considerations."
    }

    @staticmethod
    def validate_api_key(api_key: str) -> Dict[str, Any]:
        """Validate Groq API key"""
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                temperature=0
            )

            return {'valid': True, 'message': 'API key is valid'}

        except Exception as e:
            error_msg = str(e).lower()

            if 'api_key' in error_msg or 'unauthorized' in error_msg:
                return {'valid': False, 'message': 'Invalid API key'}
            elif 'model_decommissioned' in error_msg:
                return {'valid': False, 'message': 'Model decommissioned - update required'}
            else:
                return {'valid': False, 'message': f'Validation failed: {str(e)}'}

    @staticmethod
    def get_models() -> Dict[str, Any]:
        """Get available Groq models"""
        return {
            'models': GroqUtils.MODELS,
            'recommended': 'llama3-8b-8192',
            'count': len(GroqUtils.MODELS)
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    @staticmethod
    def create_medical_prompt(query: str, context: str, prompt_type: str = 'general') -> str:
        """Create medical prompt with appropriate context"""
        system_prompt = GroqUtils.SYSTEM_PROMPTS.get(prompt_type, GroqUtils.SYSTEM_PROMPTS['general'])

        return f"""{system_prompt}

Query: {query}

Medical Literature:
{context}

Provide evidence-based response addressing the query using the literature. Include limitations and actionable guidance."""

    @staticmethod
    def truncate_context(context: str, max_tokens: int, model: str = "llama3-8b-8192") -> str:
        """Truncate context to fit model limits"""
        model_limit = GroqUtils.MODELS.get(model, {}).get('context', 8192)
        actual_limit = min(max_tokens, model_limit)

        estimated_tokens = GroqUtils.estimate_tokens(context)

        if estimated_tokens <= actual_limit:
            return context

        # Calculate safe truncation length (90% of limit)
        safe_length = int(len(context) * (actual_limit / estimated_tokens) * 0.9)

        # Truncate at sentence boundary if possible
        truncated = context[:safe_length]
        last_period = truncated.rfind('. ')

        if last_period > safe_length * 0.8:  # If sentence boundary is reasonably close
            truncated = truncated[:last_period + 1]

        if len(truncated) < len(context):
            logger.warning(f"Context truncated: {len(context)} -> {len(truncated)} chars")

        return truncated.strip()

    @staticmethod
    def get_system_message(role: str) -> str:
        """Get system message for different medical roles"""
        messages = {
            'gp': "Medical AI for general practitioners. Provide comprehensive, evidence-based guidance with emphasis on clinical judgment.",
            'specialist': "Medical AI for specialist physicians. Provide detailed analysis with current literature references.",
            'nurse': "Medical AI for nursing professionals. Focus on practical, evidence-based patient care guidance.",
            'pharmacist': "Medical AI for pharmacotherapy. Specialize in medications, interactions, and pharmaceutical guidance.",
            'researcher': "Medical AI for medical research. Provide comprehensive evidence analysis with critical evaluation."
        }

        return messages.get(role, messages['gp'])

    @staticmethod
    def format_response_request(response_type: str) -> str:
        """Get response formatting instructions"""
        formats = {
            'structured': "Format as: 1) Assessment, 2) Recommendations, 3) Evidence Level, 4) Follow-up",
            'clinical': "Focus on: Differential diagnosis, treatment options, monitoring parameters, contraindications",
            'brief': "Provide concise, actionable summary with key points only",
            'detailed': "Provide comprehensive analysis with evidence levels and detailed recommendations"
        }

        return formats.get(response_type, formats['clinical'])

    @staticmethod
    def check_model_availability(model: str) -> bool:
        """Check if model is available"""
        return model in GroqUtils.MODELS

    @staticmethod
    def get_recommended_model() -> str:
        """Get recommended model for medical use"""
        return 'llama3-8b-8192'

    @staticmethod
    def calculate_cost_estimate(input_tokens: int, output_tokens: int, model: str = "llama3-8b-8192") -> Dict[
        str, float]:
        """Rough cost estimation (placeholder - update with actual Groq pricing)"""
        # Note: Update these with actual Groq pricing
        pricing = {
            "llama3-8b-8192": {"input": 0.00001, "output": 0.00002},
            "llama3-70b-8192": {"input": 0.00005, "output": 0.0001}
        }

        rates = pricing.get(model, pricing["llama3-8b-8192"])
        cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])

        return {
            "estimated_cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model
        }