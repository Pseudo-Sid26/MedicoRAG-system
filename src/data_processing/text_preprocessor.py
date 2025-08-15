import re
import time
from typing import List, Dict, Any, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import spacy

    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Using basic preprocessing.")
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    logger.info("spaCy not available. Using basic preprocessing only.")
    SPACY_AVAILABLE = False
    nlp = None


class TextPreprocessor:
    """Optimized text preprocessing for medical documents with performance focus"""

    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching
        self.nlp = nlp if SPACY_AVAILABLE else None

        # Compile regex patterns once for better performance
        self._compile_patterns()

        # Performance metrics
        self.metrics = {
            'processed_documents': 0,
            'total_processing_time': 0,
            'cache_hits': 0
        }

    def _compile_patterns(self):
        """Compile all regex patterns once for better performance"""

        # PHI patterns - compiled for speed
        self.phi_patterns = [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),  # SSN
            (re.compile(r'\b\d{3}-\d{3}-\d{4}\b'), '[PHONE]'),  # Phone
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),  # Email
            (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), '[DATE]'),  # Dates
            (re.compile(r'\b\d{4}-\d{2}-\d{2}\b'), '[DATE]'),  # ISO dates
            (re.compile(r'\bMRN[\s:#]*\d+\b', re.IGNORECASE), '[MRN]'),  # Medical record numbers
            (re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'), '[NAME]'),  # Basic name pattern
        ]

        # Text cleaning patterns
        self.cleaning_patterns = {
            'whitespace': re.compile(r'\s+'),
            'measurements': re.compile(r'(\d+)\s*(mg|ml|kg|g|l|mcg|units?|iu)\b', re.IGNORECASE),
            'temperature': re.compile(r'(\d+\.?\d*)\s*°?\s*[CF]\b'),
            'blood_pressure': re.compile(r'\b(\d{2,3})/(\d{2,3})\s*mmHg\b'),
            'ocr_errors': [
                (re.compile(r'\bl\b'), 'I'),  # Common OCR error
                (re.compile(r'\bO\b(?=\d)'), '0'),  # O instead of 0 in numbers
            ]
        }

        # Medical entity patterns
        self.medical_patterns = {
            'medications': re.compile(r'\b\w+(?:cillin|mycin|azole|pril|sartan|statin|ide|ine|ol|al|an|er)\b',
                                      re.IGNORECASE),
            'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|°[CF]|bpm|kg|cm|mm|hrs?|min|units?|iu|mcg)\b',
                                       re.IGNORECASE),
            'procedures': re.compile(
                r'\b(?:surgery|operation|procedure|intervention|biopsy|endoscopy|ultrasound|mri|ct scan|x-ray)\b',
                re.IGNORECASE)
        }

        # Medical abbreviations dictionary (essential ones only)
        self.medical_abbreviations = {
            'mg': 'milligrams', 'ml': 'milliliters', 'kg': 'kilograms',
            'bp': 'blood pressure', 'hr': 'heart rate', 'rr': 'respiratory rate',
            'dx': 'diagnosis', 'tx': 'treatment', 'rx': 'prescription',
            'hx': 'history', 'pt': 'patient', 'dr': 'doctor'
        }

    @lru_cache(maxsize=200)
    def _cached_anonymize_phi(self, text_hash: str, text: str) -> str:
        """Cache PHI anonymization for repeated content"""
        return self._anonymize_phi_internal(text)

    def _anonymize_phi_internal(self, text: str) -> str:
        """Internal PHI anonymization without caching"""
        anonymized_text = text

        # Apply all PHI patterns
        for pattern, replacement in self.phi_patterns:
            anonymized_text = pattern.sub(replacement, anonymized_text)

        return anonymized_text

    def anonymize_phi(self, text: str) -> str:
        """Remove or anonymize Personal Health Information with caching"""
        if self.enable_caching:
            text_hash = str(hash(text))
            self.metrics['cache_hits'] += 1
            return self._cached_anonymize_phi(text_hash, text)
        else:
            return self._anonymize_phi_internal(text)

    def clean_medical_text(self, text: str) -> str:
        """Optimized medical text cleaning"""
        # Remove extra whitespace
        text = self.cleaning_patterns['whitespace'].sub(' ', text)

        # Normalize medical measurements
        text = self.cleaning_patterns['measurements'].sub(r'\1 \2', text)

        # Normalize temperature readings
        text = self.cleaning_patterns['temperature'].sub(r'\1°C', text)

        # Normalize blood pressure readings
        text = self.cleaning_patterns['blood_pressure'].sub(r'\1/\2 mmHg', text)

        # Fix common OCR errors
        for pattern, replacement in self.cleaning_patterns['ocr_errors']:
            text = pattern.sub(replacement, text)

        return text.strip()

    def expand_medical_abbreviations(self, text: str) -> str:
        """Expand essential medical abbreviations only"""
        words = text.split()
        expanded_words = []

        for word in words:
            word_clean = word.lower().rstrip('.,!?;:')
            if word_clean in self.medical_abbreviations:
                expanded = self.medical_abbreviations[word_clean]
                # Preserve original case
                if word.isupper():
                    expanded = expanded.upper()
                elif word[0].isupper():
                    expanded = expanded.capitalize()
                expanded_words.append(expanded)
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Fast medical entity extraction using regex patterns"""
        entities = {
            'medications': [],
            'measurements': [],
            'procedures': []
        }

        # Extract using compiled patterns
        for entity_type, pattern in self.medical_patterns.items():
            matches = pattern.findall(text)
            entities[entity_type] = list(set(matches))  # Remove duplicates

        # Add spaCy entities if available (optional enhancement)
        if self.nlp:
            try:
                doc = self.nlp(text[:5000])  # Limit text length for performance
                spacy_entities = {
                    'persons': [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
                    'organizations': [ent.text for ent in doc.ents if ent.label_ == "ORG"],
                    'dates': [ent.text for ent in doc.ents if ent.label_ == "DATE"]
                }
                entities.update(spacy_entities)
            except Exception as e:
                logger.warning(f"spaCy processing failed: {str(e)}")

        return entities

    def segment_sentences(self, text: str) -> List[str]:
        """Fast sentence segmentation with medical context awareness"""
        if self.nlp:
            try:
                # Limit text length for performance
                doc = self.nlp(text[:10000])
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"spaCy sentence segmentation failed: {str(e)}")

        # Fallback to regex-based segmentation
        sentences = re.split(r'[.!?]+\s+', text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]

    def preprocess_document(self, content: str, anonymize: bool = True) -> Dict[str, Any]:
        """Optimized complete preprocessing pipeline"""
        start_time = time.time()

        try:
            # Step 1: Anonymize PHI if requested
            if anonymize:
                content = self.anonymize_phi(content)

            # Step 2: Clean medical text
            content = self.clean_medical_text(content)

            # Step 3: Expand essential abbreviations only
            content = self.expand_medical_abbreviations(content)

            # Step 4: Extract entities (lightweight)
            entities = self.extract_medical_entities(content)

            # Step 5: Basic sentence segmentation
            sentences = self.segment_sentences(content)

            # Step 6: Calculate basic statistics
            word_count = len(content.split())
            sentence_count = len(sentences)

            stats = {
                'total_characters': len(content),
                'total_words': word_count,
                'total_sentences': sentence_count,
                'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
            }

            processing_time = time.time() - start_time

            # Update metrics
            self.metrics['processed_documents'] += 1
            self.metrics['total_processing_time'] += processing_time

            logger.debug(f"Preprocessing completed in {processing_time:.2f}s: {stats}")

            return {
                'processed_content': content,
                'sentences': sentences,
                'entities': entities,
                'statistics': stats,
                'anonymized': anonymize,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def prepare_for_chunking(self, text: str) -> str:
        """Optimized text preparation for chunking"""
        # Ensure proper paragraph separation
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Add markers for section headers (simplified pattern)
        text = re.sub(r'^([A-Z][A-Z\s]{2,}):?\s*$', r'\n--- \1 ---\n', text, flags=re.MULTILINE)

        # Preserve medical list structures
        text = re.sub(r'\n(\d+\.|[a-z]\)|\*|\-)\s+', r'\n\1 ', text)

        return text

    def extract_clinical_evidence_indicators(self, text: str) -> Dict[str, bool]:
        """Fast evidence hierarchy detection (simplified)"""
        text_lower = text.lower()

        evidence_indicators = {
            'high_evidence': any(term in text_lower for term in [
                'systematic review', 'meta-analysis', 'cochrane', 'randomized controlled trial', 'rct'
            ]),
            'medium_evidence': any(term in text_lower for term in [
                'cohort study', 'case-control', 'clinical trial'
            ]),
            'low_evidence': any(term in text_lower for term in [
                'case series', 'case report', 'expert opinion'
            ]),
            'guidelines': any(term in text_lower for term in [
                'guideline', 'recommendation', 'consensus', 'protocol'
            ])
        }

        return evidence_indicators

    def assess_content_quality(self, text: str) -> Dict[str, Any]:
        """Fast content quality assessment"""
        word_count = len(text.split())
        sentence_count = len(self.segment_sentences(text))

        # Medical term density
        medical_matches = sum(len(pattern.findall(text)) for pattern in self.medical_patterns.values())
        medical_density = medical_matches / max(word_count, 1)

        # Structure indicators
        has_structure = bool(re.search(r'\b(abstract|introduction|methods|results|conclusion)\b', text, re.IGNORECASE))

        quality_score = 0.5  # Base score

        # Adjust based on factors
        if 100 <= word_count <= 5000:
            quality_score += 0.2
        if sentence_count >= 5:
            quality_score += 0.1
        if 0.01 <= medical_density <= 0.1:
            quality_score += 0.1
        if has_structure:
            quality_score += 0.1

        return {
            'quality_score': min(quality_score, 1.0),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'medical_density': medical_density,
            'has_structure': has_structure
        }

    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get preprocessing performance metrics"""
        avg_time = (self.metrics['total_processing_time'] /
                    max(self.metrics['processed_documents'], 1))

        return {
            'documents_processed': self.metrics['processed_documents'],
            'total_processing_time': self.metrics['total_processing_time'],
            'average_processing_time': avg_time,
            'cache_hits': self.metrics['cache_hits'],
            'spacy_available': SPACY_AVAILABLE
        }

    def clear_cache(self):
        """Clear preprocessing caches"""
        self._cached_anonymize_phi.cache_clear()
        logger.info("Text preprocessor cache cleared")

    def batch_preprocess(self, texts: List[str], anonymize: bool = True) -> List[Dict[str, Any]]:
        """Batch preprocess multiple texts efficiently"""
        results = []

        for i, text in enumerate(texts):
            try:
                result = self.preprocess_document(text, anonymize=anonymize)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to preprocess text {i}: {str(e)}")
                # Add minimal result for failed processing
                results.append({
                    'processed_content': text,
                    'sentences': [],
                    'entities': {},
                    'statistics': {'error': str(e)},
                    'anonymized': anonymize,
                    'batch_index': i,
                    'processing_failed': True
                })

        return results

    def is_medical_content(self, text: str, threshold: float = 0.3) -> bool:
        """Quick check if text contains medical content"""
        text_lower = text.lower()

        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'medication', 'clinical',
            'medical', 'therapy', 'disease', 'symptom', 'doctor', 'hospital'
        ]

        found_keywords = sum(1 for keyword in medical_keywords if keyword in text_lower)
        medical_ratio = found_keywords / len(medical_keywords)

        return medical_ratio >= threshold

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'anonymize_cache_info': self._cached_anonymize_phi.cache_info()._asdict(),
            'caching_enabled': self.enable_caching
        }