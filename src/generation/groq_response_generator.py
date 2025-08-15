import os
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class GroqResponseGenerator:
    """Generate medical responses using Groq API - OPTIMIZED VERSION"""

    # Class-level constants to avoid recreation
    SYSTEM_PROMPTS = {
        'general': """You are a medical AI assistant for healthcare professionals. Provide accurate, evidence-based medical information with citations. Emphasize clinical judgment importance. For healthcare professional use only.""",
        'diagnosis': """Medical AI assistant for diagnostic support. Analyze information and suggest evidence-based differential diagnoses. Emphasize need for clinical correlation.""",
        'treatment': """Medical AI assistant for treatment recommendations. Provide evidence-based options considering contraindications and patient factors.""",
        'drug_interaction': """Medical AI assistant for drug interactions and pharmacology. Analyze interactions, contraindications, and pharmacological considerations.""",
        'guidelines': """Medical AI assistant for clinical guidelines. Provide evidence-based recommendations following current medical standards."""
    }

    ALERT_PATTERNS = {
        'contraindication': ['contraindicated', 'should not', 'avoid'],
        'drug_interaction': ['interaction', 'concurrent use'],
        'emergency': ['emergency', 'urgent', 'immediately'],
        'monitoring': ['monitor', 'follow-up', 'check']
    }

    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            return None

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]],
                          response_type: str = 'general',
                          max_tokens: int = 1500,
                          temperature: float = 0.3) -> Dict[str, Any]:
        """Generate medical response"""

        if not self.client:
            return self._error_response('Groq client not initialized. Check API key.')

        try:
            context_text = self._prepare_context(context_chunks)
            prompt = self._create_prompt(query, context_text)

            start_time = time.time()
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": self.SYSTEM_PROMPTS.get(response_type, self.SYSTEM_PROMPTS['general'])},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            response_time = time.time() - start_time

            response_content = chat_completion.choices[0].message.content

            # Simplified response structure
            return {
                'response': response_content,
                'sources': [chunk.get('metadata', {}).get('document_source', 'Unknown') for chunk in context_chunks],
                'confidence': self._get_confidence_level(context_chunks),
                'alerts': self._get_alerts(response_content),
                'model_used': self.model,
                'response_time': response_time,
                'context_sources': len(context_chunks)
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._error_response(f'Error generating response: {str(e)}')

    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from chunks"""
        if not context_chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to top 5 chunks
            content = chunk.get('content', '')[:600]  # Truncate to 600 chars
            source = chunk.get('metadata', {}).get('document_source', 'Unknown')
            context_parts.append(f"Source {i} ({source}): {content}")

        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context_text: str) -> str:
        """Create prompt"""
        return f"""Query: {query}

Medical Literature:
{context_text}

Provide a comprehensive response addressing the query using evidence from sources. Include limitations and actionable recommendations."""

    def _get_confidence_level(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Get simple confidence level"""
        if not context_chunks:
            return 'low'

        avg_score = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
        return 'high' if avg_score > 0.8 else 'medium' if avg_score > 0.6 else 'low'

    def _get_alerts(self, response_content: str) -> List[str]:
        """Get clinical alerts"""
        alerts = []
        response_lower = response_content.lower()

        for alert_type, patterns in self.ALERT_PATTERNS.items():
            if any(pattern in response_lower for pattern in patterns):
                alerts.append(alert_type.replace('_', ' ').title())

        return alerts

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'response': message,
            'sources': [],
            'confidence': 'low',
            'alerts': [],
            'error': True
        }

    def validate_medical_query(self, query: str) -> Dict[str, Any]:
        """Validate and categorize query"""
        query_lower = query.lower()

        # Determine category
        category = 'general'
        if any(term in query_lower for term in ['diagnos', 'condition']):
            category = 'diagnosis'
        elif any(term in query_lower for term in ['treatment', 'therapy']):
            category = 'treatment'
        elif any(term in query_lower for term in ['drug', 'medication']):
            category = 'drug_interaction'

        # Check for emergency
        is_emergency = any(term in query_lower for term in ['emergency', 'urgent', 'cardiac arrest', 'stroke'])

        return {
            'category': 'emergency' if is_emergency else category,
            'is_emergency': is_emergency,
            'is_valid': True
        }