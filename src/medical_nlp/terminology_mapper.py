
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MedicalTerminologyMapper:
    """Optimized medical terminology mapper for ICD-10 and SNOMED-CT"""

    # Combined medical terms database for efficiency
    MEDICAL_TERMS = {
        # Format: term -> (icd10_code, snomed_code, category)
        'myocardial infarction': ('I21', '22298006', 'cardiovascular'),
        'heart attack': ('I21', '22298006', 'cardiovascular'),
        'mi': ('I21', '22298006', 'cardiovascular'),

        'heart failure': ('I50', '84114007', 'cardiovascular'),
        'chf': ('I50', '84114007', 'cardiovascular'),

        'hypertension': ('I10', '38341003', 'cardiovascular'),
        'high blood pressure': ('I10', '38341003', 'cardiovascular'),
        'htn': ('I10', '38341003', 'cardiovascular'),

        'diabetes mellitus': ('E11', '73211009', 'endocrine'),
        'diabetes': ('E11', '73211009', 'endocrine'),
        'dm': ('E11', '73211009', 'endocrine'),

        'asthma': ('J45', '195967001', 'respiratory'),
        'copd': ('J44', None, 'respiratory'),
        'chronic obstructive pulmonary disease': ('J44', None, 'respiratory'),

        'pneumonia': ('J18', '233604007', 'respiratory'),

        'depression': ('F32', '35489007', 'mental_health'),
        'depressive episode': ('F32', '35489007', 'mental_health'),
        'mdd': ('F32', '35489007', 'mental_health'),

        'anxiety': ('F41', None, 'mental_health'),
        'gad': ('F41', None, 'mental_health'),

        # Symptoms
        'chest pain': ('R06.02', '29857009', 'symptom'),
        'dyspnea': ('R06.0', '267036007', 'symptom'),
        'shortness of breath': ('R06.0', '267036007', 'symptom'),
        'fatigue': ('R53', '84229001', 'symptom'),
        'nausea': ('R11', '422587007', 'symptom'),
        'vomiting': ('R11', '422400008', 'symptom'),
        'headache': ('R51', '25064002', 'symptom'),
        'fever': ('R50', '386661006', 'symptom'),
        'cough': ('R05', '49727002', 'symptom')
    }

    # Abbreviation expansions
    ABBREVIATIONS = {
        'MI': 'myocardial infarction',
        'CHF': 'heart failure',
        'COPD': 'chronic obstructive pulmonary disease',
        'DM': 'diabetes mellitus',
        'HTN': 'hypertension',
        'CAD': 'coronary artery disease',
        'CVA': 'cerebrovascular accident',
        'GERD': 'gastroesophageal reflux disease',
        'RA': 'rheumatoid arthritis',
        'CKD': 'chronic kidney disease',
        'MDD': 'major depressive disorder',
        'GAD': 'generalized anxiety disorder',
        'AF': 'atrial fibrillation'
    }

    def __init__(self):
        # Pre-compile regex patterns for better performance
        self._abbrev_pattern = re.compile(r'\b(' + '|'.join(self.ABBREVIATIONS.keys()) + r')\b', re.IGNORECASE)

    def extract_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical terms from text efficiently"""
        found_terms = []
        text_lower = text.lower()

        # Find known medical terms
        for term, (icd10, snomed, category) in self.MEDICAL_TERMS.items():
            if term in text_lower:
                found_terms.append({
                    'term': term,
                    'icd10': icd10,
                    'snomed': snomed,
                    'category': category,
                    'type': 'condition' if category != 'symptom' else 'symptom'
                })

        # Find abbreviations
        for match in self._abbrev_pattern.finditer(text):
            abbrev = match.group().upper()
            if abbrev in self.ABBREVIATIONS:
                expansion = self.ABBREVIATIONS[abbrev]
                # Get codes for the expansion
                if expansion in self.MEDICAL_TERMS:
                    icd10, snomed, category = self.MEDICAL_TERMS[expansion]
                    found_terms.append({
                        'term': expansion,
                        'original': abbrev,
                        'icd10': icd10,
                        'snomed': snomed,
                        'category': category,
                        'type': 'abbreviation'
                    })

        return found_terms

    def map_to_codes(self, term: str) -> Optional[Dict[str, Any]]:
        """Map a single term to medical codes"""
        term_lower = term.lower()

        # Direct lookup
        if term_lower in self.MEDICAL_TERMS:
            icd10, snomed, category = self.MEDICAL_TERMS[term_lower]
            return {
                'term': term,
                'icd10': icd10,
                'snomed': snomed,
                'category': category
            }

        # Check if it's an abbreviation
        if term.upper() in self.ABBREVIATIONS:
            expansion = self.ABBREVIATIONS[term.upper()]
            if expansion in self.MEDICAL_TERMS:
                icd10, snomed, category = self.MEDICAL_TERMS[expansion]
                return {
                    'term': expansion,
                    'original': term,
                    'icd10': icd10,
                    'snomed': snomed,
                    'category': category
                }

        return None

    def standardize_text(self, text: str) -> Dict[str, Any]:
        """Standardize medical text by expanding abbreviations"""
        standardized_text = text
        expanded_terms = []

        # Expand abbreviations
        for match in self._abbrev_pattern.finditer(text):
            abbrev = match.group().upper()
            if abbrev in self.ABBREVIATIONS:
                expansion = self.ABBREVIATIONS[abbrev]
                standardized_text = standardized_text.replace(match.group(), expansion)
                expanded_terms.append({
                    'original': abbrev,
                    'expansion': expansion
                })

        return {
            'original_text': text,
            'standardized_text': standardized_text,
            'expanded_terms': expanded_terms
        }

    def analyze_medical_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis of medical text"""
        # Extract terms and standardize
        terms = self.extract_medical_terms(text)
        standardized = self.standardize_text(text)

        # Group by category and coding system
        by_category = {}
        icd10_codes = []
        snomed_codes = []

        for term in terms:
            category = term['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(term)

            if term['icd10']:
                icd10_codes.append({
                    'code': term['icd10'],
                    'term': term['term'],
                    'category': category
                })

            if term['snomed']:
                snomed_codes.append({
                    'code': term['snomed'],
                    'term': term['term'],
                    'category': category
                })

        return {
            'terms_found': terms,
            'total_terms': len(terms),
            'by_category': by_category,
            'icd10_codes': icd10_codes,
            'snomed_codes': snomed_codes,
            'standardized_text': standardized['standardized_text'],
            'coverage_score': min(len(terms) / max(len(text.split()), 1), 1.0)
        }

    def suggest_codes(self, description: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Suggest medical codes based on description"""
        suggestions = []
        desc_words = set(description.lower().split())

        for term, (icd10, snomed, category) in self.MEDICAL_TERMS.items():
            term_words = set(term.split())

            # Calculate word overlap score
            overlap = len(desc_words.intersection(term_words))
            if overlap > 0:
                total_words = len(desc_words.union(term_words))
                score = overlap / total_words

                suggestion = {
                    'term': term,
                    'category': category,
                    'relevance_score': score
                }

                if icd10:
                    suggestion['icd10'] = icd10
                if snomed:
                    suggestion['snomed'] = snomed

                suggestions.append(suggestion)

        # Sort by relevance and return top suggestions
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        return suggestions[:max_suggestions]

    def validate_codes(self, codes: List[str], system: str = 'ICD-10') -> Dict[str, Any]:
        """Validate medical codes"""
        valid_codes = []
        invalid_codes = []

        # Create lookup set for efficient validation
        if system == 'ICD-10':
            valid_set = {icd10 for icd10, _, _ in self.MEDICAL_TERMS.values() if icd10}
        elif system == 'SNOMED-CT':
            valid_set = {snomed for _, snomed, _ in self.MEDICAL_TERMS.values() if snomed}
        else:
            return {'error': f'Unsupported system: {system}'}

        for code in codes:
            if code in valid_set:
                # Find the term for this code
                term_info = self._find_term_by_code(code, system)
                valid_codes.append(term_info)
            else:
                invalid_codes.append(code)

        return {
            'system': system,
            'valid_codes': valid_codes,
            'invalid_codes': invalid_codes,
            'validation_rate': len(valid_codes) / len(codes) if codes else 0
        }

    def _find_term_by_code(self, code: str, system: str) -> Dict[str, Any]:
        """Find term information by code"""
        for term, (icd10, snomed, category) in self.MEDICAL_TERMS.items():
            if system == 'ICD-10' and icd10 == code:
                return {'code': code, 'term': term, 'category': category}
            elif system == 'SNOMED-CT' and snomed == code:
                return {'code': code, 'term': term, 'category': category}
        return {'code': code, 'term': 'Unknown', 'category': 'unknown'}

    def get_term_info(self, term: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a medical term"""
        mapping = self.map_to_codes(term)
        if mapping:
            return {
                'term': mapping['term'],
                'category': mapping['category'],
                'codes': {
                    'icd10': mapping.get('icd10'),
                    'snomed': mapping.get('snomed')
                },
                'is_abbreviation': 'original' in mapping
            }
        return None

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts efficiently"""
        results = []
        for text in texts:
            try:
                analysis = self.analyze_medical_text(text)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'analysis': analysis,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'error': str(e),
                    'success': False
                })
        return results

    def get_category_stats(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Get statistics by medical category"""
        stats = {}
        for category, terms in analysis.get('by_category', {}).items():
            stats[category] = len(terms)
        return stats