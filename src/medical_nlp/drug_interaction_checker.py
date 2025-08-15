

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