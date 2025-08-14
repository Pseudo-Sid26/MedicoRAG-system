# import time
# import logging
# from typing import Dict, Any
#
# logger = logging.getLogger(__name__)
#
# class GroqUtils:
#     """Utility functions for Groq API operations"""
#
#     @staticmethod
#     def validate_api_key(api_key: str) -> Dict[str, Any]:
#         """Validate Groq API key with current models"""
#         try:
#             from groq import Groq
#
#             # Create client
#             client = Groq(api_key=api_key)
#
#             # Test with current available model
#             response = client.chat.completions.create(
#                 model="llama3-8b-8192",  # Current working model
#                 messages=[{"role": "user", "content": "Test"}],
#                 max_tokens=5,
#                 temperature=0
#             )
#
#             return {
#                 'valid': True,
#                 'message': 'API key is valid',
#                 'model_access': True
#             }
#
#         except Exception as e:
#             error_msg = str(e)
#             logger.error(f"API validation error: {error_msg}")
#
#             if 'api_key' in error_msg.lower() or 'unauthorized' in error_msg.lower():
#                 return {
#                     'valid': False,
#                     'message': 'Invalid API key - check your Groq API key',
#                     'error': error_msg
#                 }
#             elif 'model_decommissioned' in error_msg.lower():
#                 return {
#                     'valid': False,
#                     'message': 'Model decommissioned - updating to current model',
#                     'error': error_msg
#                 }
#             else:
#                 return {
#                     'valid': False,
#                     'message': f'API validation failed: {error_msg}',
#                     'error': error_msg
#                 }
#
#     @staticmethod
#     def get_available_models(api_key: str) -> Dict[str, Any]:
#         """Get list of current available Groq models (as of August 2025)"""
#         try:
#             # Current Groq models available
#             available_models = {
#                 "llama3-8b-8192": {
#                     "name": "Llama 3 8B",
#                     "context_length": 8192,
#                     "description": "Meta's Llama 3 model - fast and efficient",
#                     "use_case": "General purpose, good for medical tasks"
#                 },
#                 "llama3-70b-8192": {
#                     "name": "Llama 3 70B",
#                     "context_length": 8192,
#                     "description": "Larger Llama 3 model - more capable",
#                     "use_case": "Complex reasoning, detailed analysis"
#                 },
#                 "mixtral-8x7b-32768": {
#                     "name": "Mixtral 8x7B (DEPRECATED)",
#                     "context_length": 32768,
#                     "description": "DEPRECATED - use Llama 3 models instead",
#                     "use_case": "DO NOT USE - model decommissioned"
#                 },
#                 "gemma-7b-it": {
#                     "name": "Gemma 7B IT",
#                     "context_length": 8192,
#                     "description": "Google's Gemma instruction-tuned model",
#                     "use_case": "Instruction following, reasoning"
#                 }
#             }
#
#             return {
#                 'success': True,
#                 'models': available_models,
#                 'count': len(available_models),
#                 'recommended': 'llama3-8b-8192'
#             }
#
#         except Exception as e:
#             return {
#                 'success': False,
#                 'error': str(e),
#                 'models': {}
#             }
#
#     @staticmethod
#     def estimate_tokens(text: str) -> int:
#         """Estimate token count for text (rough approximation)"""
#         return len(text) // 4
#
#     @staticmethod
#     def format_medical_prompt(query: str, context: str, prompt_type: str = 'general') -> str:
#         """Format prompt for medical use cases"""
#
#         medical_disclaimers = {
#             'general': "Important: This is for healthcare professional use only. Always apply clinical judgment.",
#             'diagnosis': "Note: This is diagnostic support only. Clinical correlation and additional testing may be required.",
#             'treatment': "Important: Consider patient-specific factors, allergies, and contraindications before implementing any treatment.",
#             'drug_interaction': "Critical: Verify all drug interactions with current references and consider patient-specific factors."
#         }
#
#         disclaimer = medical_disclaimers.get(prompt_type, medical_disclaimers['general'])
#
#         formatted_prompt = f"""
# {disclaimer}
#
# Medical Query: {query}
#
# Relevant Medical Literature and Context:
# {context}
#
# Please provide an evidence-based response that:
# 1. Directly addresses the medical query
# 2. References relevant evidence from the provided context
# 3. Acknowledges any limitations or uncertainties
# 4. Provides actionable clinical guidance when appropriate
# 5. Maintains appropriate medical terminology while being clear
# """
#
#         return formatted_prompt
#
#     @staticmethod
#     def optimize_context_length(context: str, max_tokens: int, model: str) -> str:
#         """Optimize context length for model limits"""
#
#         # Updated model limits for current models
#         model_limits = {
#             "llama3-8b-8192": 8192,
#             "llama3-70b-8192": 8192,
#             "gemma-7b-it": 8192,
#             "mixtral-8x7b-32768": 32768  # Keep for backwards compatibility
#         }
#
#         limit = model_limits.get(model, 8192)
#         estimated_tokens = GroqUtils.estimate_tokens(context)
#
#         if estimated_tokens <= max_tokens:
#             return context
#
#         # Calculate truncation point
#         truncate_ratio = max_tokens / estimated_tokens
#         truncate_length = int(len(context) * truncate_ratio * 0.9)  # 90% to be safe
#
#         # Try to truncate at sentence boundaries
#         sentences = context.split('. ')
#         truncated_context = ""
#
#         for sentence in sentences:
#             if len(truncated_context) + len(sentence) + 2 <= truncate_length:
#                 truncated_context += sentence + ". "
#             else:
#                 break
#
#         logger.warning(f"Context truncated from {len(context)} to {len(truncated_context)} characters")
#         return truncated_context.strip()
#
#     @staticmethod
#     def create_system_message(role_type: str) -> str:
#         """Create appropriate system message for different medical roles"""
#
#         system_messages = {
#             'general_practitioner': """You are a medical AI assistant helping general practitioners with evidence-based clinical decisions.
#             Provide comprehensive, accurate medical information while emphasizing the importance of clinical judgment and patient-specific factors.""",
#
#             'specialist': """You are a medical AI assistant supporting specialist physicians with advanced clinical analysis.
#             Provide detailed, evidence-based recommendations with appropriate references to current medical literature.""",
#
#             'nurse': """You are a medical AI assistant supporting nursing professionals with patient care decisions.
#             Focus on practical, evidence-based guidance for nursing interventions and patient monitoring.""",
#
#             'pharmacist': """You are a medical AI assistant specializing in pharmacotherapy and drug interactions.
#             Provide detailed medication analysis, interaction checking, and pharmaceutical guidance.""",
#
#             'researcher': """You are a medical AI assistant supporting medical research and literature analysis.
#             Provide comprehensive analysis of medical evidence with appropriate critical evaluation."""
#         }
#
#         return system_messages.get(role_type, system_messages['general_practitioner'])


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