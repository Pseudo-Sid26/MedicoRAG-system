# import re
# import json
# from typing import List, Dict, Any, Optional, Tuple
# import requests
# import logging
# from dataclasses import dataclass
#
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class DrugInteraction:
#     drug1: str
#     drug2: str
#     severity: str
#     description: str
#     mechanism: str
#     management: str
#     evidence_level: str
#
#
# @dataclass
# class Drug:
#     name: str
#     generic_name: str
#     brand_names: List[str]
#     drug_class: str
#     mechanism: str
#     contraindications: List[str]
#     warnings: List[str]
#
#
# class DrugInteractionChecker:
#     """Advanced drug interaction and contraindication checker"""
#
#     def __init__(self):
#         self.drug_database = {}
#         self.interaction_database = {}
#         self.load_drug_database()
#         self.load_interaction_patterns()
#
#     def load_drug_database(self):
#         """Load comprehensive drug database"""
#         # Common drug classes and their interactions
#         self.drug_classes = {
#             'ace_inhibitors': {
#                 'drugs': ['lisinopril', 'enalapril', 'captopril', 'ramipril'],
#                 'interactions': ['nsaids', 'potassium_supplements', 'arbs'],
#                 'contraindications': ['pregnancy', 'bilateral_renal_artery_stenosis']
#             },
#             'beta_blockers': {
#                 'drugs': ['metoprolol', 'atenolol', 'propranolol', 'carvedilol'],
#                 'interactions': ['calcium_channel_blockers', 'insulin', 'epinephrine'],
#                 'contraindications': ['asthma', 'severe_bradycardia', 'heart_block']
#             },
#             'anticoagulants': {
#                 'drugs': ['warfarin', 'heparin', 'rivaroxaban', 'apixaban', 'dabigatran'],
#                 'interactions': ['nsaids', 'antibiotics', 'antifungals', 'antiplatelet'],
#                 'contraindications': ['active_bleeding', 'severe_hepatic_impairment']
#             },
#             'nsaids': {
#                 'drugs': ['ibuprofen', 'naproxen', 'diclofenac', 'celecoxib'],
#                 'interactions': ['ace_inhibitors', 'anticoagulants', 'lithium', 'methotrexate'],
#                 'contraindications': ['peptic_ulcer', 'severe_heart_failure', 'severe_renal_impairment']
#             },
#             'statins': {
#                 'drugs': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin'],
#                 'interactions': ['macrolides', 'azole_antifungals', 'cyclosporine'],
#                 'contraindications': ['active_liver_disease', 'pregnancy', 'breastfeeding']
#             },
#             'antibiotics_macrolides': {
#                 'drugs': ['azithromycin', 'clarithromycin', 'erythromycin'],
#                 'interactions': ['statins', 'warfarin', 'digoxin', 'theophylline'],
#                 'contraindications': ['qtc_prolongation', 'severe_hepatic_impairment']
#             }
#         }
#
#         # Specific drug information
#         self.specific_drugs = {
#             'warfarin': {
#                 'class': 'anticoagulant',
#                 'mechanism': 'vitamin_k_antagonist',
#                 'major_interactions': ['antibiotics', 'antifungals', 'nsaids', 'amiodarone'],
#                 'monitoring': 'inr',
#                 'reversal_agent': 'vitamin_k'
#             },
#             'digoxin': {
#                 'class': 'cardiac_glycoside',
#                 'mechanism': 'na_k_atpase_inhibitor',
#                 'major_interactions': ['amiodarone', 'verapamil', 'quinidine', 'macrolides'],
#                 'monitoring': 'serum_level',
#                 'toxicity_signs': ['nausea', 'confusion', 'arrhythmias']
#             },
#             'lithium': {
#                 'class': 'mood_stabilizer',
#                 'mechanism': 'unknown',
#                 'major_interactions': ['nsaids', 'ace_inhibitors', 'thiazides', 'metronidazole'],
#                 'monitoring': 'serum_level',
#                 'toxicity_signs': ['tremor', 'confusion', 'seizures']
#             }
#         }
#
#     def load_interaction_patterns(self):
#         """Load drug interaction patterns and severity classifications"""
#         self.interaction_patterns = {
#             'major': {
#                 'warfarin + nsaids': {
#                     'description': 'Increased bleeding risk',
#                     'mechanism': 'Additive anticoagulant effects and gastric irritation',
#                     'management': 'Avoid combination or use with extreme caution. Consider gastroprotection.',
#                     'monitoring': 'INR, signs of bleeding'
#                 },
#                 'ace_inhibitors + potassium': {
#                     'description': 'Hyperkalemia risk',
#                     'mechanism': 'Reduced potassium excretion',
#                     'management': 'Monitor serum potassium levels closely',
#                     'monitoring': 'Serum potassium, renal function'
#                 },
#                 'statins + macrolides': {
#                     'description': 'Increased statin toxicity risk',
#                     'mechanism': 'CYP3A4 inhibition increases statin levels',
#                     'management': 'Consider statin suspension or dose reduction',
#                     'monitoring': 'Muscle symptoms, CK levels'
#                 }
#             },
#             'moderate': {
#                 'beta_blockers + calcium_channel_blockers': {
#                     'description': 'Additive negative inotropic effects',
#                     'mechanism': 'Combined negative effects on heart rate and contractility',
#                     'management': 'Monitor heart rate and blood pressure',
#                     'monitoring': 'Heart rate, blood pressure, signs of heart failure'
#                 }
#             },
#             'minor': {
#                 'antacids + antibiotics': {
#                     'description': 'Reduced antibiotic absorption',
#                     'mechanism': 'Chelation or pH changes',
#                     'management': 'Separate administration by 2-4 hours',
#                     'monitoring': 'Clinical response to antibiotic'
#                 }
#             }
#         }
#
#     def extract_drugs_from_text(self, text: str) -> List[Dict[str, Any]]:
#         """Extract drug names from text using pattern recognition"""
#         found_drugs = []
#         text_lower = text.lower()
#
#         # Check for specific drugs
#         for drug_name, info in self.specific_drugs.items():
#             if drug_name in text_lower:
#                 found_drugs.append({
#                     'name': drug_name,
#                     'type': 'specific',
#                     'class': info['class'],
#                     'mechanism': info['mechanism']
#                 })
#
#         # Check for drug classes
#         for class_name, class_info in self.drug_classes.items():
#             for drug in class_info['drugs']:
#                 if drug in text_lower:
#                     found_drugs.append({
#                         'name': drug,
#                         'type': 'class_member',
#                         'class': class_name,
#                         'mechanism': 'unknown'
#                     })
#
#         # Pattern-based drug detection
#         drug_patterns = [
#             r'\b\w+cillin\b',  # Penicillins
#             r'\b\w+mycin\b',  # Mycins
#             r'\b\w+pril\b',  # ACE inhibitors
#             r'\b\w+sartan\b',  # ARBs
#             r'\b\w+statin\b',  # Statins
#             r'\b\w+zole\b',  # Azoles
#             r'\b\w+olol\b'  # Beta blockers
#         ]
#
#         for pattern in drug_patterns:
#             matches = re.findall(pattern, text_lower)
#             for match in matches:
#                 if not any(drug['name'] == match for drug in found_drugs):
#                     found_drugs.append({
#                         'name': match,
#                         'type': 'pattern_detected',
#                         'class': 'unknown',
#                         'mechanism': 'unknown'
#                     })
#
#         return found_drugs
#
#     def check_drug_interactions(self, drugs: List[str]) -> List[DrugInteraction]:
#         """Check for interactions between multiple drugs"""
#         interactions = []
#
#         # Check all drug pairs
#         for i in range(len(drugs)):
#             for j in range(i + 1, len(drugs)):
#                 drug1, drug2 = drugs[i].lower(), drugs[j].lower()
#                 interaction = self.check_pair_interaction(drug1, drug2)
#                 if interaction:
#                     interactions.append(interaction)
#
#         return interactions
#
#     def check_pair_interaction(self, drug1: str, drug2: str) -> Optional[DrugInteraction]:
#         """Check interaction between two specific drugs"""
#         # Normalize drug names
#         drug1, drug2 = drug1.lower(), drug2.lower()
#
#         # Check direct patterns
#         pair_key = f"{drug1} + {drug2}"
#         reverse_key = f"{drug2} + {drug1}"
#
#         for severity, interactions in self.interaction_patterns.items():
#             if pair_key in interactions:
#                 info = interactions[pair_key]
#                 return DrugInteraction(
#                     drug1=drug1,
#                     drug2=drug2,
#                     severity=severity,
#                     description=info['description'],
#                     mechanism=info['mechanism'],
#                     management=info['management'],
#                     evidence_level='established'
#                 )
#             elif reverse_key in interactions:
#                 info = interactions[reverse_key]
#                 return DrugInteraction(
#                     drug1=drug2,
#                     drug2=drug1,
#                     severity=severity,
#                     description=info['description'],
#                     mechanism=info['mechanism'],
#                     management=info['management'],
#                     evidence_level='established'
#                 )
#
#         # Check class-based interactions
#         class_interaction = self.check_class_interaction(drug1, drug2)
#         if class_interaction:
#             return class_interaction
#
#         return None
#
#     def check_class_interaction(self, drug1: str, drug2: str) -> Optional[DrugInteraction]:
#         """Check for class-based drug interactions"""
#         drug1_class = self.get_drug_class(drug1)
#         drug2_class = self.get_drug_class(drug2)
#
#         if not drug1_class or not drug2_class:
#             return None
#
#         # Check if classes interact
#         class1_info = self.drug_classes.get(drug1_class, {})
#         class2_info = self.drug_classes.get(drug2_class, {})
#
#         if drug2_class in class1_info.get('interactions', []):
#             return DrugInteraction(
#                 drug1=drug1,
#                 drug2=drug2,
#                 severity='moderate',
#                 description=f'Class interaction between {drug1_class} and {drug2_class}',
#                 mechanism='Class-based interaction pattern',
#                 management='Monitor for enhanced effects or toxicity',
#                 evidence_level='theoretical'
#             )
#
#         return None
#
#     def get_drug_class(self, drug_name: str) -> Optional[str]:
#         """Get drug class for a given drug name"""
#         drug_name = drug_name.lower()
#
#         # Check specific drugs first
#         if drug_name in self.specific_drugs:
#             return self.specific_drugs[drug_name]['class']
#
#         # Check drug classes
#         for class_name, class_info in self.drug_classes.items():
#             if drug_name in class_info['drugs']:
#                 return class_name
#
#         return None
#
#     def check_contraindications(self, drug: str, patient_conditions: List[str]) -> List[Dict[str, Any]]:
#         """Check for contraindications based on patient conditions"""
#         contraindications = []
#         drug = drug.lower()
#
#         drug_class = self.get_drug_class(drug)
#         if not drug_class:
#             return contraindications
#
#         class_info = self.drug_classes.get(drug_class, {})
#         drug_contraindications = class_info.get('contraindications', [])
#
#         for condition in patient_conditions:
#             condition = condition.lower().replace(' ', '_')
#             if condition in drug_contraindications:
#                 contraindications.append({
#                     'drug': drug,
#                     'condition': condition,
#                     'severity': 'absolute' if 'severe' in condition else 'relative',
#                     'recommendation': f'Avoid {drug} in patients with {condition.replace("_", " ")}'
#                 })
#
#         return contraindications
#
#     def analyze_medication_list(self, medication_text: str, patient_conditions: List[str] = None) -> Dict[str, Any]:
#         """Comprehensive analysis of a medication list"""
#         # Extract drugs
#         drugs = self.extract_drugs_from_text(medication_text)
#         drug_names = [drug['name'] for drug in drugs]
#
#         # Check interactions
#         interactions = self.check_drug_interactions(drug_names)
#
#         # Check contraindications if patient conditions provided
#         contraindications = []
#         if patient_conditions:
#             for drug_name in drug_names:
#                 drug_contraindications = self.check_contraindications(drug_name, patient_conditions)
#                 contraindications.extend(drug_contraindications)
#
#         # Generate recommendations
#         recommendations = self.generate_recommendations(interactions, contraindications)
#
#         return {
#             'drugs_found': drugs,
#             'total_drugs': len(drugs),
#             'interactions': [
#                 {
#                     'drug1': interaction.drug1,
#                     'drug2': interaction.drug2,
#                     'severity': interaction.severity,
#                     'description': interaction.description,
#                     'mechanism': interaction.mechanism,
#                     'management': interaction.management
#                 } for interaction in interactions
#             ],
#             'contraindications': contraindications,
#             'recommendations': recommendations,
#             'risk_assessment': self.assess_overall_risk(interactions, contraindications)
#         }
#
#     def generate_recommendations(self, interactions: List[DrugInteraction],
#                                  contraindications: List[Dict[str, Any]]) -> List[str]:
#         """Generate clinical recommendations based on findings"""
#         recommendations = []
#
#         # Interaction-based recommendations
#         major_interactions = [i for i in interactions if i.severity == 'major']
#         if major_interactions:
#             recommendations.append("URGENT: Major drug interactions detected - review immediately")
#             for interaction in major_interactions:
#                 recommendations.append(f"- {interaction.drug1} + {interaction.drug2}: {interaction.management}")
#
#         moderate_interactions = [i for i in interactions if i.severity == 'moderate']
#         if moderate_interactions:
#             recommendations.append("Monitor for moderate drug interactions")
#
#         # Contraindication-based recommendations
#         absolute_contraindications = [c for c in contraindications if c['severity'] == 'absolute']
#         if absolute_contraindications:
#             recommendations.append("ALERT: Absolute contraindications identified")
#             for contra in absolute_contraindications:
#                 recommendations.append(f"- {contra['recommendation']}")
#
#         # General recommendations
#         if interactions or contraindications:
#             recommendations.extend([
#                 "Consider medication reconciliation",
#                 "Review with clinical pharmacist",
#                 "Monitor patient closely for adverse effects",
#                 "Document clinical decision-making rationale"
#             ])
#
#         return recommendations
#
#     def assess_overall_risk(self, interactions: List[DrugInteraction],
#                             contraindications: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Assess overall medication risk"""
#         risk_score = 0
#         risk_factors = []
#
#         # Score interactions
#         for interaction in interactions:
#             if interaction.severity == 'major':
#                 risk_score += 3
#                 risk_factors.append(f"Major interaction: {interaction.drug1} + {interaction.drug2}")
#             elif interaction.severity == 'moderate':
#                 risk_score += 2
#                 risk_factors.append(f"Moderate interaction: {interaction.drug1} + {interaction.drug2}")
#             else:
#                 risk_score += 1
#
#         # Score contraindications
#         for contra in contraindications:
#             if contra['severity'] == 'absolute':
#                 risk_score += 4
#                 risk_factors.append(f"Absolute contraindication: {contra['drug']}")
#             else:
#                 risk_score += 2
#                 risk_factors.append(f"Relative contraindication: {contra['drug']}")
#
#         # Determine risk level
#         if risk_score >= 8:
#             risk_level = 'very_high'
#             recommendation = 'Immediate medication review required'
#         elif risk_score >= 5:
#             risk_level = 'high'
#             recommendation = 'Urgent medication review recommended'
#         elif risk_score >= 3:
#             risk_level = 'moderate'
#             recommendation = 'Medication review advisable'
#         elif risk_score >= 1:
#             risk_level = 'low'
#             recommendation = 'Monitor patient response'
#         else:
#             risk_level = 'minimal'
#             recommendation = 'No immediate concerns identified'
#
#         return {
#             'risk_level': risk_level,
#             'risk_score': risk_score,
#             'risk_factors': risk_factors,
#             'recommendation': recommendation
#         }
#
#     def check_contraindications_with_conditions(self, medications: List[str],
#                                                 patient_conditions: List[str],
#                                                 patient_age: str = None) -> List[Dict]:
#         """Enhanced contraindication checking with patient-specific factors"""
#         contraindications = []
#
#         # Age-specific contraindications
#         age_contraindications = {
#             'elderly': {
#                 'beers_criteria': ['diphenhydramine', 'amitriptyline', 'diazepam'],
#                 'fall_risk': ['sedatives', 'hypnotics', 'antipsychotics']
#             },
#             'pediatric': {
#                 'not_recommended': ['aspirin', 'tetracycline', 'fluoroquinolones']
#             }
#         }
#
#         # Enhanced condition-specific contraindications
#         condition_contraindications = {
#             'kidney_disease': ['nsaids', 'metformin', 'ace_inhibitors'],
#             'liver_disease': ['acetaminophen', 'statins', 'antifungals'],
#             'heart_failure': ['nsaids', 'thiazolidinediones'],
#             'pregnancy': ['ace_inhibitors', 'warfarin', 'statins']
#         }
#
#         for medication in medications:
#             for condition in patient_conditions:
#                 if condition.lower().replace(' ', '_') in condition_contraindications:
#                     contraindicated_drugs = condition_contraindications[condition.lower().replace(' ', '_')]
#                     if any(drug in medication.lower() for drug in contraindicated_drugs):
#                         contraindications.append({
#                             'medication': medication,
#                             'condition': condition,
#                             'severity': 'major',
#                             'recommendation': f'Avoid {medication} in patients with {condition}',
#                             'evidence_level': 'A'
#                         })
#
#         return contraindications
#
#     def assess_medication_appropriateness(self, medications: List[str],
#                                           patient_data: Dict) -> Dict[str, Any]:
#         """Assess medication appropriateness using clinical criteria"""
#         assessment = {
#             'appropriate': [],
#             'questionable': [],
#             'inappropriate': [],
#             'missing_indications': [],
#             'dose_adjustments_needed': []
#         }
#
#         # Check for duplicate therapy
#         drug_classes = self._categorize_by_drug_class(medications)
#         for drug_class, drugs in drug_classes.items():
#             if len(drugs) > 1:
#                 assessment['questionable'].append({
#                     'issue': 'duplicate_therapy',
#                     'medications': drugs,
#                     'recommendation': f'Review need for multiple {drug_class} medications'
#                 })
#
#         return assessment


import re
from typing import List, Dict, Any, Optional, Set
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DrugInteraction:
    drug1: str
    drug2: str
    severity: str
    description: str
    management: str


class DrugInteractionChecker:
    """Optimized drug interaction and contraindication checker"""

    # Class-level constants for better performance
    DRUG_PATTERNS = [
        (r'\b\w+cillin\b', 'antibiotic'),
        (r'\b\w+mycin\b', 'antibiotic'),
        (r'\b\w+pril\b', 'ace_inhibitor'),
        (r'\b\w+sartan\b', 'arb'),
        (r'\b\w+statin\b', 'statin'),
        (r'\b\w+zole\b', 'antifungal'),
        (r'\b\w+olol\b', 'beta_blocker')
    ]

    # Simplified drug database
    DRUGS = {
        # ACE Inhibitors
        'lisinopril': 'ace_inhibitor', 'enalapril': 'ace_inhibitor', 'captopril': 'ace_inhibitor',
        # Beta Blockers
        'metoprolol': 'beta_blocker', 'atenolol': 'beta_blocker', 'propranolol': 'beta_blocker',
        # Anticoagulants
        'warfarin': 'anticoagulant', 'rivaroxaban': 'anticoagulant', 'apixaban': 'anticoagulant',
        # NSAIDs
        'ibuprofen': 'nsaid', 'naproxen': 'nsaid', 'diclofenac': 'nsaid',
        # Statins
        'atorvastatin': 'statin', 'simvastatin': 'statin', 'rosuvastatin': 'statin',
        # Antibiotics
        'azithromycin': 'macrolide', 'clarithromycin': 'macrolide', 'erythromycin': 'macrolide',
        # Special monitoring drugs
        'digoxin': 'cardiac_glycoside', 'lithium': 'mood_stabilizer'
    }

    # Major interactions only (most clinically significant)
    MAJOR_INTERACTIONS = {
        ('anticoagulant', 'nsaid'): {
            'description': 'Increased bleeding risk',
            'management': 'Avoid combination or use gastroprotection'
        },
        ('ace_inhibitor', 'nsaid'): {
            'description': 'Reduced ACE inhibitor effect, kidney damage risk',
            'management': 'Monitor kidney function and blood pressure'
        },
        ('statin', 'macrolide'): {
            'description': 'Increased statin toxicity risk',
            'management': 'Consider statin suspension or dose reduction'
        },
        ('anticoagulant', 'macrolide'): {
            'description': 'Increased anticoagulation effect',
            'management': 'Monitor INR closely'
        }
    }

    # Contraindications by drug class
    CONTRAINDICATIONS = {
        'ace_inhibitor': ['pregnancy', 'bilateral_renal_artery_stenosis'],
        'beta_blocker': ['asthma', 'severe_bradycardia'],
        'anticoagulant': ['active_bleeding', 'severe_hepatic_impairment'],
        'nsaid': ['peptic_ulcer', 'severe_heart_failure'],
        'statin': ['active_liver_disease', 'pregnancy']
    }

    def __init__(self):
        # Compile regex patterns once
        self._compiled_patterns = [(re.compile(pattern, re.IGNORECASE), drug_class)
                                   for pattern, drug_class in self.DRUG_PATTERNS]

    def extract_drugs_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract drugs from text efficiently"""
        found_drugs = []
        text_lower = text.lower()
        found_names = set()

        # Check known drugs first (fastest)
        for drug_name, drug_class in self.DRUGS.items():
            if drug_name in text_lower and drug_name not in found_names:
                found_drugs.append({'name': drug_name, 'class': drug_class})
                found_names.add(drug_name)

        # Check patterns for unknown drugs
        for pattern, drug_class in self._compiled_patterns:
            for match in pattern.finditer(text):
                drug_name = match.group().lower()
                if drug_name not in found_names:
                    found_drugs.append({'name': drug_name, 'class': drug_class})
                    found_names.add(drug_name)

        return found_drugs

    def check_interactions(self, drugs: List[Dict[str, str]]) -> List[DrugInteraction]:
        """Check for major drug interactions"""
        interactions = []

        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                drug1, drug2 = drugs[i], drugs[j]

                # Check class interaction
                class_pair = tuple(sorted([drug1['class'], drug2['class']]))

                if class_pair in self.MAJOR_INTERACTIONS:
                    info = self.MAJOR_INTERACTIONS[class_pair]
                    interactions.append(DrugInteraction(
                        drug1=drug1['name'],
                        drug2=drug2['name'],
                        severity='major',
                        description=info['description'],
                        management=info['management']
                    ))

        return interactions

    def check_contraindications(self, drugs: List[Dict[str, str]],
                                conditions: List[str]) -> List[Dict[str, str]]:
        """Check contraindications efficiently"""
        contraindications = []
        conditions_set = {c.lower().replace(' ', '_') for c in conditions}

        for drug in drugs:
            drug_class = drug['class']
            if drug_class in self.CONTRAINDICATIONS:
                for contraindication in self.CONTRAINDICATIONS[drug_class]:
                    if contraindication in conditions_set:
                        contraindications.append({
                            'drug': drug['name'],
                            'condition': contraindication.replace('_', ' '),
                            'severity': 'major'
                        })

        return contraindications

    def analyze_medication_list(self, medication_text: str,
                                patient_conditions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main analysis function"""
        # Extract drugs
        drugs = self.extract_drugs_from_text(medication_text)

        # Check interactions
        interactions = self.check_interactions(drugs)

        # Check contraindications
        contraindications = []
        if patient_conditions:
            contraindications = self.check_contraindications(drugs, patient_conditions)

        # Assess risk
        risk_assessment = self._assess_risk(interactions, contraindications)

        return {
            'drugs_found': drugs,
            'total_drugs': len(drugs),
            'interactions': [{
                'drug1': i.drug1,
                'drug2': i.drug2,
                'severity': i.severity,
                'description': i.description,
                'management': i.management
            } for i in interactions],
            'contraindications': contraindications,
            'recommendations': self._generate_recommendations(interactions, contraindications),
            'risk_assessment': risk_assessment
        }

    def _assess_risk(self, interactions: List[DrugInteraction],
                     contraindications: List[Dict[str, str]]) -> Dict[str, Any]:
        """Simple risk assessment"""
        risk_score = len(interactions) * 3 + len(contraindications) * 4

        if risk_score >= 8:
            level, recommendation = 'very_high', 'Immediate review required'
        elif risk_score >= 5:
            level, recommendation = 'high', 'Urgent review recommended'
        elif risk_score >= 3:
            level, recommendation = 'moderate', 'Review advisable'
        elif risk_score >= 1:
            level, recommendation = 'low', 'Monitor patient'
        else:
            level, recommendation = 'minimal', 'No immediate concerns'

        return {
            'risk_level': level,
            'risk_score': risk_score,
            'recommendation': recommendation,
            'total_issues': len(interactions) + len(contraindications)
        }

    def _generate_recommendations(self, interactions: List[DrugInteraction],
                                  contraindications: List[Dict[str, str]]) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []

        if interactions:
            recommendations.append("Major drug interactions detected - review immediately")
            for interaction in interactions[:3]:  # Limit to top 3
                recommendations.append(f"{interaction.drug1} + {interaction.drug2}: {interaction.management}")

        if contraindications:
            recommendations.append("Contraindications identified")
            for contra in contraindications[:3]:  # Limit to top 3
                recommendations.append(f"Avoid {contra['drug']} with {contra['condition']}")

        if interactions or contraindications:
            recommendations.extend([
                "Consider medication reconciliation",
                "Review with clinical pharmacist"
            ])

        return recommendations

    def quick_check(self, drug1: str, drug2: str) -> Optional[str]:
        """Quick interaction check between two drugs"""
        class1 = self.DRUGS.get(drug1.lower())
        class2 = self.DRUGS.get(drug2.lower())

        if not class1 or not class2:
            return None

        class_pair = tuple(sorted([class1, class2]))
        if class_pair in self.MAJOR_INTERACTIONS:
            return self.MAJOR_INTERACTIONS[class_pair]['description']

        return None

    def get_drug_info(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get basic drug information"""
        drug_class = self.DRUGS.get(drug_name.lower())
        if not drug_class:
            return None

        return {
            'name': drug_name,
            'class': drug_class,
            'contraindications': self.CONTRAINDICATIONS.get(drug_class, [])
        }