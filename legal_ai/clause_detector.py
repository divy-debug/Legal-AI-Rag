from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from parser import ParsedDocument

logger = logging.getLogger(__name__)


class ClauseType(str, Enum):
    INDEMNITY = "indemnity"
    LIABILITY = "liability"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    PAYMENT = "payment"
    ARBITRATION = "arbitration"


class DetectedClause(BaseModel):
    clause_type: ClauseType
    matched_text: str = Field()
    trigger_pattern: str = Field()
    element_index: int = Field()
    page_number: int | None = Field(default=None)
    section_heading: str | None = Field(default=None)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


CLAUSE_KEYWORDS: dict[ClauseType, list[str]] = {
    ClauseType.INDEMNITY: [
        "indemnify",
        "indemnification",
        "indemnity",
        "hold harmless",
        "save harmless",
        "defend and indemnify",
    ],
    ClauseType.LIABILITY: [
        "limitation of liability",
        "liable",
        "liability",
        "damages",
        "consequential damages",
        "direct damages",
        "aggregate liability",
        "exclusion of liability",
        "cap on liability",
    ],
    ClauseType.TERMINATION: [
        "terminate",
        "termination",
        "cancellation",
        "expiration",
        "right to terminate",
        "termination for cause",
        "termination for convenience",
        "notice of termination",
        "survival",
    ],
    ClauseType.NON_COMPETE: [
        "non-compete",
        "non compete",
        "non-competition",
        "non competition",
        "non-solicitation",
        "non solicitation",
        "restrictive covenant",
        "non-disclosure",
        "confidentiality",
        "trade secret",
        "exclusivity",
    ],
    ClauseType.PAYMENT: [
        "payment terms",
        "invoice",
        "net ",
        "compensation",
        "fees",
        "reimbursement",
        "late payment",
        "penalty",
        "interest rate",
        "milestone payment",
    ],
    ClauseType.ARBITRATION: [
        "arbitration",
        "dispute resolution",
        "mediation",
        "governing law",
        "jurisdiction",
        "venue",
        "choice of law",
        "binding arbitration",
        "forum selection",
    ],
}


def _split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+|\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _compute_confidence(match_count: int) -> float:
    if match_count <= 0:
        return 0.0
    base = 0.6
    bonus = 0.1 * (match_count - 1)
    return min(base + bonus, 1.0)


def _pick_best_sentence(sentences: list[str], keywords: list[str]) -> str:
    if not sentences:
        return ""

    lowered_keywords = [k.lower() for k in keywords]

    for sentence in sentences:
        lower = sentence.lower()
        if any(k in lower for k in lowered_keywords):
            return sentence

    return sentences[0]


class ClauseDetector:
    def detect(self, parsed_doc: ParsedDocument) -> list[DetectedClause]:
        from parser import ElementType

        detected: list[DetectedClause] = []

        for element in parsed_doc.elements:
            if element.element_type == ElementType.IMAGE:
                continue

            text = (element.content or "").strip()
            if len(text) < 10:
                continue

            sentences = _split_into_sentences(text) or [text]
            lower_text = text.lower()

            for clause_type, keywords in CLAUSE_KEYWORDS.items():
                matched = [k for k in keywords if k.lower() in lower_text]
                if not matched:
                    continue

                best_sentence = _pick_best_sentence(sentences, matched)
                if len(best_sentence) > 500:
                    best_sentence = best_sentence[:497] + "..."

                confidence = _compute_confidence(len(matched))
                trigger = ", ".join(sorted(set(matched)))

                detected.append(
                    DetectedClause(
                        clause_type=clause_type,
                        matched_text=best_sentence,
                        trigger_pattern=trigger,
                        element_index=element.element_index,
                        page_number=element.page_number,
                        section_heading=element.section_heading,
                        confidence=round(confidence, 2),
                    )
                )

        logger.info(
            "Detected %d clause(s) across %d type(s) in %s",
            len(detected),
            len({c.clause_type for c in detected}),
            parsed_doc.source_file,
        )
        return detected
