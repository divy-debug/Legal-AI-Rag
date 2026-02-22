"""
Legal AI package — clause detection and risk classification.
"""

from legal_ai.clause_detector import ClauseDetector, ClauseType, DetectedClause
from legal_ai.risk_classifier import RiskAssessment, RiskClassifier, RiskLevel

__all__ = [
    "ClauseDetector",
    "ClauseType",
    "DetectedClause",
    "RiskAssessment",
    "RiskClassifier",
    "RiskLevel",
]
