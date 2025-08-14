# import re
# import json
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class MedicalTerm:
#     term: str
#     code: str
#     system: str  # ICD-10, SNOMED, etc.
#     category: str
#     description: str
#     synonyms: List[str]
#
#
# class MedicalTerminologyMapper:
#     """Maps medical terms to standardized coding systems (ICD-10, SNOMED-CT)"""
#
#     def __init__(self):
#         self.icd10_codes = {}
#         self.snomed_codes = {}
#         self.medical_abbreviations = {}
#         self.anatomy_terms = {}
#         self.symptom_terms = {}
#         self.procedure_terms = {}
#
#         self.load_terminology_databases()
#
#     def load_terminology_databases(self):
#         """Load medical terminology databases"""
#         # ICD-10 common codes (subset for demonstration)
#         self.icd10_codes = {
#             # Cardiovascular
#             'I21': {'description': 'Acute myocardial infarction', 'category': 'cardiovascular'},
#             'I50': {'description': 'Heart failure', 'category': 'cardiovascular'},
#             'I25': {'description': 'Chronic ischemic heart disease', 'category': 'cardiovascular'},
#             'I10': {'description': 'Essential hypertension', 'category': 'cardiovascular'},
#             'I48': {'description': 'Atrial fibrillation and flutter', 'category': 'cardiovascular'},
#
#             # Respiratory
#             'J44': {'description': 'Chronic obstructive pulmonary disease', 'category': 'respiratory'},
#             'J45': {'description': 'Asthma', 'category': 'respiratory'},
#             'J18': {'description': 'Pneumonia', 'category': 'respiratory'},
#             'J06': {'description': 'Acute upper respiratory infections', 'category': 'respiratory'},
#
#             # Diabetes and Endocrine
#             'E11': {'description': 'Type 2 diabetes mellitus', 'category': 'endocrine'},
#             'E10': {'description': 'Type 1 diabetes mellitus', 'category': 'endocrine'},
#             'E78': {'description': 'Disorders of lipoprotein metabolism', 'category': 'endocrine'},
#
#             # Mental Health
#             'F32': {'description': 'Depressive episode', 'category': 'mental_health'},
#             'F41': {'description': 'Anxiety disorders', 'category': 'mental_health'},
#
#             # Gastrointestinal
#             'K21': {'description': 'Gastro-esophageal reflux disease', 'category': 'gastrointestinal'},
#             'K59': {'description': 'Other functional intestinal disorders', 'category': 'gastrointestinal'},
#
#             # Musculoskeletal
#             'M54': {'description': 'Dorsalgia', 'category': 'musculoskeletal'},
#             'M25': {'description': 'Other joint disorders', 'category': 'musculoskeletal'}
#         }
#
#         # SNOMED-CT concepts (subset)
#         self.snomed_codes = {
#             '22298006': {'description': 'Myocardial infarction', 'category': 'finding'},
#             '84114007': {'description': 'Heart failure', 'category': 'finding'},
#             '38341003': {'description': 'Hypertensive disorder', 'category': 'finding'},
#             '195967001': {'description': 'Asthma', 'category': 'finding'},
#             '44054006': {'description': 'Diabetes mellitus type 2', 'category': 'finding'},
#             '73211009': {'description': 'Diabetes mellitus', 'category': 'finding'},
#             '233604007': {'description': 'Pneumonia', 'category': 'finding'},
#             '35489007': {'description': 'Depressive disorder', 'category': 'finding'}
#         }
#
#         # Medical abbreviations and their expansions
#         self.medical_abbreviations = {
#             'MI': 'myocardial infarction',
#             'CHF': 'congestive heart failure',
#             'COPD': 'chronic obstructive pulmonary disease',
#             'DM': 'diabetes mellitus',
#             'HTN': 'hypertension',
#             'CAD': 'coronary artery disease',
#             'CVA': 'cerebrovascular accident',
#             'DVT': 'deep vein thrombosis',
#             'PE': 'pulmonary embolism',
#             'UTI': 'urinary tract infection',
#             'GERD': 'gastroesophageal reflux disease',
#             'RA': 'rheumatoid arthritis',
#             'OA': 'osteoarthritis',
#             'CKD': 'chronic kidney disease',
#             'ESRD': 'end-stage renal disease',
#             'PTSD': 'post-traumatic stress disorder',
#             'MDD': 'major depressive disorder',
#             'GAD': 'generalized anxiety disorder',
#             'AF': 'atrial fibrillation',
#             'VT': 'ventricular tachycardia',
#             'SVT': 'supraventricular tachycardia'
#         }
#
#         # Anatomical terms
#         self.anatomy_terms = {
#             'cardiac': 'heart',
#             'pulmonary': 'lung',
#             'renal': 'kidney',
#             'hepatic': 'liver',
#             'cerebral': 'brain',
#             'gastric': 'stomach',
#             'intestinal': 'intestine',
#             'vascular': 'blood vessel',
#             'muscular': 'muscle',
#             'skeletal': 'bone',
#             'dermal': 'skin',
#             'ocular': 'eye',
#             'auditory': 'ear',
#             'nasal': 'nose',
#             'oral': 'mouth'
#         }
#
#         # Common symptoms and their standardized terms
#         self.symptom_terms = {
#             'chest pain': {'icd10': 'R06.02', 'snomed': '29857009'},
#             'shortness of breath': {'icd10': 'R06.02', 'snomed': '267036007'},
#             'dyspnea': {'icd10': 'R06.0', 'snomed': '267036007'},
#             'fatigue': {'icd10': 'R53', 'snomed': '84229001'},
#             'nausea': {'icd10': 'R11', 'snomed': '422587007'},
#             'vomiting': {'icd10': 'R11', 'snomed': '422400008'},
#             'diarrhea': {'icd10': 'K59.1', 'snomed': '62315008'},
#             'constipation': {'icd10': 'K59.0', 'snomed': '14760008'},
#             'headache': {'icd10': 'R51', 'snomed': '25064002'},
#             'dizziness': {'icd10': 'R42', 'snomed': '404640003'},
#             'fever': {'icd10': 'R50', 'snomed': '386661006'},
#             'cough': {'icd10': 'R05', 'snomed': '49727002'},
#             'palpitations': {'icd10': 'R00.2', 'snomed': '80313002'}
#         }
#
#         # Medical procedures
#         self.procedure_terms = {
#             'echocardiogram': {'cpt': '93306', 'snomed': '40701008'},
#             'electrocardiogram': {'cpt': '93000', 'snomed': '29303009'},
#             'chest x-ray': {'cpt': '71045', 'snomed': '399208008'},
#             'blood test': {'cpt': '80053', 'snomed': '396550006'},
#             'urinalysis': {'cpt': '81001', 'snomed': '167217005'},
#             'colonoscopy': {'cpt': '45378', 'snomed': '73761001'},
#             'endoscopy': {'cpt': '43235', 'snomed': '423827005'},
#             'biopsy': {'cpt': '88305', 'snomed': '86273004'},
#             'mri': {'cpt': '70551', 'snomed': '113091000'},
#             'ct scan': {'cpt': '74150', 'snomed': '77477000'}
#         }
#
#     def extract_medical_terms(self, text: str) -> List[Dict[str, Any]]:
#         """Extract and categorize medical terms from text"""
#         found_terms = []
#         text_lower = text.lower()
#
#         # Find abbreviations
#         for abbrev, expansion in self.medical_abbreviations.items():
#             pattern = r'\b' + re.escape(abbrev.lower()) + r'\b'
#             if re.search(pattern, text_lower):
#                 found_terms.append({
#                     'original': abbrev,
#                     'standardized': expansion,
#                     'type': 'abbreviation',
#                     'category': 'condition'
#                 })
#
#         # Find symptoms
#         for symptom, codes in self.symptom_terms.items():
#             if symptom in text_lower:
#                 found_terms.append({
#                     'original': symptom,
#                     'standardized': symptom,
#                     'type': 'symptom',
#                     'category': 'finding',
#                     'codes': codes
#                 })
#
#         # Find procedures
#         for procedure, codes in self.procedure_terms.items():
#             if procedure in text_lower:
#                 found_terms.append({
#                     'original': procedure,
#                     'standardized': procedure,
#                     'type': 'procedure',
#                     'category': 'procedure',
#                     'codes': codes
#                 })
#
#         # Find anatomical terms
#         for term, standardized in self.anatomy_terms.items():
#             if term in text_lower:
#                 found_terms.append({
#                     'original': term,
#                     'standardized': standardized,
#                     'type': 'anatomy',
#                     'category': 'body_structure'
#                 })
#
#         return found_terms
#
#     def map_to_icd10(self, condition: str) -> Optional[Dict[str, Any]]:
#         """Map a condition to ICD-10 code"""
#         condition_lower = condition.lower()
#
#         # Direct mapping
#         for code, info in self.icd10_codes.items():
#             if condition_lower in info['description'].lower():
#                 return {
#                     'code': code,
#                     'description': info['description'],
#                     'category': info['category'],
#                     'system': 'ICD-10'
#                 }
#
#         # Check abbreviations
#         if condition.upper() in self.medical_abbreviations:
#             expanded = self.medical_abbreviations[condition.upper()]
#             return self.map_to_icd10(expanded)
#
#         # Pattern matching for common conditions
#         condition_patterns = {
#             r'heart attack|myocardial infarction|mi': 'I21',
#             r'heart failure|chf': 'I50',
#             r'diabetes|dm': 'E11',
#             r'hypertension|high blood pressure|htn': 'I10',
#             r'asthma': 'J45',
#             r'copd|chronic obstructive': 'J44',
#             r'pneumonia': 'J18',
#             r'depression|depressive': 'F32',
#             r'anxiety': 'F41'
#         }
#
#         for pattern, code in condition_patterns.items():
#             if re.search(pattern, condition_lower):
#                 if code in self.icd10_codes:
#                     info = self.icd10_codes[code]
#                     return {
#                         'code': code,
#                         'description': info['description'],
#                         'category': info['category'],
#                         'system': 'ICD-10'
#                     }
#
#         return None
#
#     def map_to_snomed(self, term: str) -> Optional[Dict[str, Any]]:
#         """Map a term to SNOMED-CT code"""
#         term_lower = term.lower()
#
#         # Direct mapping
#         for code, info in self.snomed_codes.items():
#             if term_lower in info['description'].lower():
#                 return {
#                     'code': code,
#                     'description': info['description'],
#                     'category': info['category'],
#                     'system': 'SNOMED-CT'
#                 }
#
#         # Pattern matching
#         snomed_patterns = {
#             r'heart attack|myocardial infarction': '22298006',
#             r'heart failure': '84114007',
#             r'hypertension|high blood pressure': '38341003',
#             r'asthma': '195967001',
#             r'diabetes.*type 2': '44054006',
#             r'diabetes': '73211009',
#             r'pneumonia': '233604007',
#             r'depression|depressive': '35489007'
#         }
#
#         for pattern, code in snomed_patterns.items():
#             if re.search(pattern, term_lower):
#                 if code in self.snomed_codes:
#                     info = self.snomed_codes[code]
#                     return {
#                         'code': code,
#                         'description': info['description'],
#                         'category': info['category'],
#                         'system': 'SNOMED-CT'
#                     }
#
#         return None
#
#     def standardize_medical_text(self, text: str) -> Dict[str, Any]:
#         """Standardize medical text with terminology mapping"""
#         # Extract terms
#         extracted_terms = self.extract_medical_terms(text)
#
#         # Expand abbreviations
#         standardized_text = text
#         for abbrev, expansion in self.medical_abbreviations.items():
#             pattern = r'\b' + re.escape(abbrev) + r'\b'
#             standardized_text = re.sub(pattern, expansion, standardized_text, flags=re.IGNORECASE)
#
#         # Map to coding systems
#         mapped_codes = []
#         for term in extracted_terms:
#             if term['type'] in ['abbreviation', 'symptom']:
#                 # Try ICD-10 mapping
#                 icd_mapping = self.map_to_icd10(term['standardized'])
#                 if icd_mapping:
#                     mapped_codes.append(icd_mapping)
#
#                 # Try SNOMED mapping
#                 snomed_mapping = self.map_to_snomed(term['standardized'])
#                 if snomed_mapping:
#                     mapped_codes.append(snomed_mapping)
#
#         return {
#             'original_text': text,
#             'standardized_text': standardized_text,
#             'extracted_terms': extracted_terms,
#             'mapped_codes': mapped_codes,
#             'terminology_coverage': len(mapped_codes) / max(len(extracted_terms), 1)
#         }
#
#     def validate_medical_coding(self, codes: List[str], system: str = 'ICD-10') -> Dict[str, Any]:
#         """Validate medical codes against known databases"""
#         valid_codes = []
#         invalid_codes = []
#
#         if system == 'ICD-10':
#             reference_db = self.icd10_codes
#         elif system == 'SNOMED-CT':
#             reference_db = self.snomed_codes
#         else:
#             return {'error': f'Unsupported coding system: {system}'}
#
#         for code in codes:
#             if code in reference_db:
#                 valid_codes.append({
#                     'code': code,
#                     'description': reference_db[code]['description'],
#                     'category': reference_db[code]['category']
#                 })
#             else:
#                 invalid_codes.append(code)
#
#         return {
#             'system': system,
#             'total_codes': len(codes),
#             'valid_codes': valid_codes,
#             'invalid_codes': invalid_codes,
#             'validation_rate': len(valid_codes) / len(codes) if codes else 0
#         }
#
#     def suggest_codes(self, description: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
#         """Suggest relevant medical codes based on description"""
#         suggestions = []
#         description_lower = description.lower()
#
#         # Search ICD-10 codes
#         for code, info in self.icd10_codes.items():
#             desc_lower = info['description'].lower()
#             # Simple relevance scoring based on word overlap
#             desc_words = set(desc_lower.split())
#             input_words = set(description_lower.split())
#             overlap = len(desc_words.intersection(input_words))
#
#             if overlap > 0:
#                 suggestions.append({
#                     'code': code,
#                     'description': info['description'],
#                     'category': info['category'],
#                     'system': 'ICD-10',
#                     'relevance_score': overlap / len(desc_words.union(input_words))
#                 })
#
#         # Search SNOMED codes
#         for code, info in self.snomed_codes.items():
#             desc_lower = info['description'].lower()
#             desc_words = set(desc_lower.split())
#             input_words = set(description_lower.split())
#             overlap = len(desc_words.intersection(input_words))
#
#             if overlap > 0:
#                 suggestions.append({
#                     'code': code,
#                     'description': info['description'],
#                     'category': info['category'],
#                     'system': 'SNOMED-CT',
#                     'relevance_score': overlap / len(desc_words.union(input_words))
#                 })
#
#         # Sort by relevance and return top suggestions
#         suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
#         return suggestions[:max_suggestions]
#
#     def create_coding_summary(self, text: str) -> Dict[str, Any]:
#         """Create a comprehensive coding summary for medical text"""
#         standardized = self.standardize_medical_text(text)
#
#         # Group codes by system
#         codes_by_system = {}
#         for code in standardized['mapped_codes']:
#             system = code['system']
#             if system not in codes_by_system:
#                 codes_by_system[system] = []
#             codes_by_system[system].append(code)
#
#         # Generate statistics
#         stats = {
#             'total_terms_found': len(standardized['extracted_terms']),
#             'total_codes_mapped': len(standardized['mapped_codes']),
#             'systems_used': list(codes_by_system.keys()),
#             'mapping_coverage': standardized['terminology_coverage']
#         }
#
#         return {
#             'text_analysis': standardized,
#             'codes_by_system': codes_by_system,
#             'statistics': stats,
#             'recommendations': self._generate_coding_recommendations(standardized)
#         }
#
#     def _generate_coding_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
#         """Generate recommendations for medical coding"""
#         recommendations = []
#
#         if analysis['terminology_coverage'] < 0.5:
#             recommendations.append("Consider manual review - low terminology coverage detected")
#
#         if len(analysis['mapped_codes']) == 0:
#             recommendations.append("No standard codes identified - manual coding may be required")
#
#         unmapped_terms = [
#             term for term in analysis['extracted_terms']
#             if not any(code.get('original_term') == term['original'] for code in analysis['mapped_codes'])
#         ]
#
#         if unmapped_terms:
#             recommendations.append(f"Review {len(unmapped_terms)} unmapped medical terms")
#
#         if any(term['type'] == 'abbreviation' for term in analysis['extracted_terms']):
#             recommendations.append("Verify abbreviation expansions for accuracy")
#
#         return recommendations
#
#     def map_to_clinical_guidelines(self, text: str) -> List[Dict]:
#         """Map content to relevant clinical guidelines"""
#         guideline_mapping = {
#             'diabetes': {
#                 'ada_guidelines': 'American Diabetes Association Standards',
#                 'aace_guidelines': 'AACE Diabetes Guidelines'
#             },
#             'hypertension': {
#                 'acc_aha': 'ACC/AHA Hypertension Guidelines',
#                 'esh_esc': 'ESH/ESC Hypertension Guidelines'
#             },
#             'heart_failure': {
#                 'acc_aha_hfsa': 'ACC/AHA/HFSA Heart Failure Guidelines'
#             }
#         }
#
#         relevant_guidelines = []
#         text_lower = text.lower()
#
#         for condition, guidelines in guideline_mapping.items():
#             if condition in text_lower:
#                 for guideline_key, guideline_name in guidelines.items():
#                     relevant_guidelines.append({
#                         'condition': condition,
#                         'guideline': guideline_name,
#                         'relevance_score': text_lower.count(condition) / len(text.split()) * 100
#                     })
#
#         return relevant_guidelines
#
#     def validate_medical_coding_accuracy(self, extracted_codes: List[str],
#                                          document_content: str) -> Dict[str, Any]:
#         """Validate accuracy of extracted medical codes"""
#         validation_results = {
#             'accurate_codes': [],
#             'questionable_codes': [],
#             'missing_codes': [],
#             'confidence_scores': {}
#         }
#
#         # Cross-reference codes with content
#         for code in extracted_codes:
#             if code in self.icd10_codes:
#                 code_term = self.icd10_codes[code]['description']
#                 if code_term.lower() in document_content.lower():
#                     validation_results['accurate_codes'].append(code)
#                     validation_results['confidence_scores'][code] = 0.9
#                 else:
#                     validation_results['questionable_codes'].append(code)
#                     validation_results['confidence_scores'][code] = 0.3
#
#         return validation_results


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