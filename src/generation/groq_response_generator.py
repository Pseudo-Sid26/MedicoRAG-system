# import os
# import time
# import logging
# from typing import List, Dict, Any, Optional
#
# logger = logging.getLogger(__name__)
#
# class GroqResponseGenerator:
#     """Generate medical responses using Groq API - SIMPLIFIED VERSION"""
#
#     def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
#         self.api_key = api_key
#         self.model = model
#         self.client = None
#         self._initialize_client()
#
#         # Medical response templates
#         self.system_prompts = {
#             'general': """You are a medical AI assistant designed to help healthcare professionals make evidence-based decisions.
#             You provide accurate, evidence-based medical information while emphasizing the importance of clinical judgment and patient-specific factors.
#             Always include relevant citations and acknowledge limitations in your responses.
#             Important: This is for healthcare professional use only and should not replace clinical judgment.""",
#
#             'diagnosis': """You are a medical AI assistant specializing in diagnostic support.
#             Analyze the provided medical information and suggest potential diagnoses based on evidence.
#             Present differential diagnoses with supporting evidence, but emphasize that clinical correlation and further testing may be needed.""",
#
#             'treatment': """You are a medical AI assistant specializing in treatment recommendations.
#             Provide evidence-based treatment options while considering contraindications, drug interactions, and patient factors.""",
#
#             'drug_interaction': """You are a medical AI assistant specializing in drug interactions and pharmacology.
#             Analyze potential drug interactions, contraindications, and pharmacological considerations.""",
#
#             'guidelines': """You are a medical AI assistant specializing in clinical guidelines and best practices.
#             Provide evidence-based recommendations following current medical guidelines and standards of care."""
#         }
#
#     def _initialize_client(self):
#         """Initialize Groq client with minimal configuration"""
#         try:
#             from groq import Groq
#             # Create client with absolute minimal configuration
#             self.client = Groq(api_key=self.api_key)
#             logger.info("Groq client initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize Groq client: {str(e)}")
#             self.client = None
#
#     def generate_response(self, query: str, context_chunks: List[Dict[str, Any]],
#                          response_type: str = 'general',
#                          max_tokens: int = 1500,
#                          temperature: float = 0.3) -> Dict[str, Any]:
#         """Generate a comprehensive medical response"""
#
#         if not self.client:
#             return {
#                 'response': {
#                     'main_response': 'Error: Groq client not properly initialized. Please check your API key.',
#                     'sources_used': [],
#                     'confidence_assessment': {'level': 'low', 'reason': 'Client error'},
#                     'clinical_alerts': [],
#                     'follow_up_suggestions': ['Check API key configuration'],
#                     'evidence_level': {'level': 'insufficient', 'description': 'No API access'}
#                 },
#                 'error': True
#             }
#
#         try:
#             # Prepare context
#             context_text = self._prepare_context(context_chunks)
#
#             # Create the prompt
#             prompt = self._create_prompt(query, context_text, response_type)
#
#             # Generate response
#             start_time = time.time()
#
#             chat_completion = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": self.system_prompts.get(response_type, self.system_prompts['general'])},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=temperature
#             )
#
#             response_time = time.time() - start_time
#
#             # Extract response
#             response_content = chat_completion.choices[0].message.content
#
#             # Create enhanced response
#             enhanced_response = {
#                 'main_response': response_content,
#                 'sources_used': self._process_sources(context_chunks),
#                 'confidence_assessment': self._assess_confidence(context_chunks),
#                 'clinical_alerts': self._identify_clinical_alerts(response_content, query),
#                 'follow_up_suggestions': self._generate_follow_up_suggestions(response_content, response_type),
#                 'evidence_level': self._assess_evidence_level(context_chunks)
#             }
#
#             return {
#                 'response': enhanced_response,
#                 'model_used': self.model,
#                 'response_time': response_time,
#                 'tokens_used': getattr(chat_completion.usage, 'total_tokens', None) if hasattr(chat_completion, 'usage') else None,
#                 'context_sources': len(context_chunks),
#                 'query_type': response_type
#             }
#
#         except Exception as e:
#             logger.error(f"Error generating response: {str(e)}")
#             return {
#                 'response': {
#                     'main_response': f'Error generating response: {str(e)}',
#                     'sources_used': [],
#                     'confidence_assessment': {'level': 'low', 'reason': 'Generation error'},
#                     'clinical_alerts': [],
#                     'follow_up_suggestions': ['Try again or check system status'],
#                     'evidence_level': {'level': 'insufficient', 'description': 'Error occurred'}
#                 },
#                 'error': True
#             }
#
#     def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
#         """Prepare context from retrieved chunks"""
#         if not context_chunks:
#             return "No relevant context found."
#
#         context_parts = []
#         for i, chunk in enumerate(context_chunks, 1):
#             content = chunk.get('content', '')
#             metadata = chunk.get('metadata', {})
#
#             context_entry = f"--- Source {i} ---\n"
#             context_entry += f"Document: {metadata.get('document_source', 'Unknown')}\n"
#             context_entry += f"Section: {metadata.get('section', 'Unknown')}\n"
#             context_entry += f"Content: {content[:800]}...\n" if len(content) > 800 else f"Content: {content}\n"
#
#             context_parts.append(context_entry)
#
#         return "\n".join(context_parts)
#
#     def _create_prompt(self, query: str, context_text: str, response_type: str) -> str:
#         """Create appropriate prompt based on response type"""
#         return f"""Based on the following medical literature, please provide a comprehensive response to the query.
#
# QUERY: {query}
#
# RELEVANT MEDICAL LITERATURE:
# {context_text}
#
# Please provide a response that:
# 1. Directly addresses the query
# 2. Uses evidence from the provided sources
# 3. Includes appropriate medical context
# 4. Acknowledges limitations and uncertainties
# 5. Provides actionable recommendations when appropriate"""
#
#     def _process_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Process sources for response"""
#         sources = []
#         for chunk in context_chunks:
#             metadata = chunk.get('metadata', {})
#             sources.append({
#                 'document': metadata.get('document_source', 'Unknown'),
#                 'section': metadata.get('section', 'Unknown'),
#                 'relevance_score': chunk.get('similarity_score', 0),
#                 'contains_diagnosis': metadata.get('contains_diagnosis', False),
#                 'contains_treatment': metadata.get('contains_treatment', False),
#                 'clinical_relevance': metadata.get('clinical_relevance', 0)
#             })
#         return sources
#
#     def _assess_confidence(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Assess confidence in the response"""
#         if not context_chunks:
#             return {'level': 'low', 'reason': 'No relevant sources found'}
#
#         avg_relevance = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
#
#         if avg_relevance > 0.8:
#             level = 'high'
#             reason = f'High relevance sources (avg: {avg_relevance:.2f})'
#         elif avg_relevance > 0.6:
#             level = 'medium'
#             reason = f'Moderate relevance sources (avg: {avg_relevance:.2f})'
#         else:
#             level = 'low'
#             reason = f'Limited relevance sources (avg: {avg_relevance:.2f})'
#
#         return {'level': level, 'reason': reason}
#
#     def _identify_clinical_alerts(self, response_content: str, query: str) -> List[Dict[str, str]]:
#         """Identify potential clinical alerts"""
#         alerts = []
#         response_lower = response_content.lower()
#
#         alert_patterns = {
#             'contraindication': ['contraindicated', 'should not', 'avoid'],
#             'drug_interaction': ['interaction', 'concurrent use'],
#             'emergency': ['emergency', 'urgent', 'immediately'],
#             'monitoring': ['monitor', 'follow-up', 'check']
#         }
#
#         for alert_type, patterns in alert_patterns.items():
#             for pattern in patterns:
#                 if pattern in response_lower:
#                     alerts.append({
#                         'type': alert_type,
#                         'message': f'Response contains {alert_type} information'
#                     })
#                     break
#
#         return alerts
#
#     def _generate_follow_up_suggestions(self, response_content: str, response_type: str) -> List[str]:
#         """Generate follow-up suggestions"""
#         base_suggestions = [
#             "Consult current clinical guidelines",
#             "Consider patient-specific factors",
#             "Review with supervising physician if uncertain"
#         ]
#
#         type_specific = {
#             'diagnosis': ["Consider additional diagnostic tests"],
#             'treatment': ["Verify dosing and contraindications"],
#             'drug_interaction': ["Consult pharmacy for verification"]
#         }
#
#         suggestions = base_suggestions.copy()
#         suggestions.extend(type_specific.get(response_type, []))
#         return suggestions
#
#     def _assess_evidence_level(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Assess evidence level"""
#         if not context_chunks:
#             return {'level': 'insufficient', 'description': 'No sources available'}
#
#         return {
#             'level': 'moderate',
#             'description': f'Based on {len(context_chunks)} literature sources',
#             'source_distribution': {'medical_literature': len(context_chunks)}
#         }
#
#     def validate_medical_query(self, query: str) -> Dict[str, Any]:
#         """Validate medical query"""
#         validation = {
#             'is_valid': True,
#             'category': 'general',
#             'warnings': [],
#             'suggestions': []
#         }
#
#         # Emergency indicators
#         emergency_terms = ['emergency', 'urgent', 'cardiac arrest', 'stroke']
#         if any(term in query.lower() for term in emergency_terms):
#             validation['warnings'].append('Emergency situation - seek immediate medical attention')
#             validation['category'] = 'emergency'
#
#         # Determine category
#         if any(term in query.lower() for term in ['diagnos', 'condition']):
#             validation['category'] = 'diagnosis'
#         elif any(term in query.lower() for term in ['treatment', 'therapy']):
#             validation['category'] = 'treatment'
#         elif any(term in query.lower() for term in ['drug', 'medication']):
#             validation['category'] = 'drug_interaction'
#
#         return validation


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