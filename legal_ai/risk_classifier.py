
from __future__ import annotations

import logging
import re
from enum import Enum

from pydantic import BaseModel, Field

from legal_ai.clause_detector import ClauseType, DetectedClause
from llm import classify_clause_risk

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskAssessment(BaseModel):
    """Risk assessment for a single detected clause."""

    clause_type: ClauseType
    risk_level: RiskLevel
    explanation: str = Field(
        description="Human-readable explanation of why this risk level was assigned"
    )
    element_index: int
    page_number: int | None = None
    section_heading: str | None = None
    matched_text: str = ""


import time

# ... (other imports)

class RiskClassifier:
    """Classify detected clauses by risk level using OpenAI."""

    def classify(self, clauses: list[DetectedClause]) -> list[RiskAssessment]:
        """Classify each detected clause as HIGH, MEDIUM, or LOW risk."""
        assessments: list[RiskAssessment] = []

        for i, clause in enumerate(clauses):
            if i > 0:
                time.sleep(2.0) # More delay between LLM calls for risk analysis
            level_str, explanation = classify_clause_risk(
                clause_type=clause.clause_type,
                clause_text=clause.matched_text,
            )
            level = {
                "High": RiskLevel.HIGH,
                "Medium": RiskLevel.MEDIUM,
                "Low": RiskLevel.LOW,
            }.get(level_str, RiskLevel.MEDIUM)

            assessments.append(
                RiskAssessment(
                    clause_type=clause.clause_type,
                    risk_level=level,
                    explanation=explanation,
                    element_index=clause.element_index,
                    page_number=clause.page_number,
                    section_heading=clause.section_heading,
                    matched_text=clause.matched_text,
                )
            )

        high = sum(1 for a in assessments if a.risk_level == RiskLevel.HIGH)
        med = sum(1 for a in assessments if a.risk_level == RiskLevel.MEDIUM)
        low = sum(1 for a in assessments if a.risk_level == RiskLevel.LOW)
        logger.info("Risk breakdown: %d HIGH, %d MEDIUM, %d LOW", high, med, low)

        return assessments
