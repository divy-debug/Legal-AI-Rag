from __future__ import annotations

import logging
import os
import json
import time
from typing import TYPE_CHECKING, Iterable, Sequence, Any

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

from legal_ai.clause_detector import ClauseType

if TYPE_CHECKING:
    from utils import CitationChunk

load_dotenv()

logger = logging.getLogger(__name__)

# Clients
_OPENAI_CLIENT: OpenAI | None = None
_GEMINI_MODEL: genai.GenerativeModel | None = None

def _get_provider() -> str:
    return os.getenv("AI_PROVIDER", "openai").lower()

def _openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT

def _gemini_model() -> genai.GenerativeModel:
    global _GEMINI_MODEL
    if _GEMINI_MODEL is None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        _GEMINI_MODEL = genai.GenerativeModel(model_name)
    return _GEMINI_MODEL

def _gemini_generate_with_retry(model_instance, prompt, max_retries=5):
    """Simple retry logic for Gemini quota issues."""
    for attempt in range(max_retries):
        try:
            return model_instance.generate_content(prompt)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1) # Wait longer: 5s, 10s, 15s, 20s, 25s
                    logger.warning(f"Gemini quota hit, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            raise e
    return None

def generate_image_caption(
    *,
    page_number: int | None,
    section_heading: str | None,
    model: str | None = None,
) -> str:
    if os.getenv("USE_MOCK_MODELS", "false").lower() == "true":
        return "Image on page " + str(page_number or "?")

    page_text = f"page {page_number}" if page_number is not None else "the document"
    heading = section_heading or "an unlabeled section"
    prompt = (
        f"Generate a short, neutral legal document image caption (max 12 words) for an image on {page_text} "
        f"under section '{heading}'. Use generic wording."
    )

    if _get_provider() == "gemini":
        try:
            model_instance = _gemini_model()
            response = _gemini_generate_with_retry(model_instance, prompt)
            return response.text.strip()
        except Exception:
            return "Embedded figure in legal document"
    else:
        client = _openai_client()
        model_name = model or os.getenv("OPENAI_IMAGE_CAPTION_MODEL", "gpt-4.1-mini")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
        )
        return response.choices[0].message.content.strip()

def classify_clause_risk(
    clause_type: ClauseType,
    clause_text: str,
    model: str | None = None,
) -> tuple[str, str]:
    if os.getenv("USE_MOCK_MODELS", "false").lower() == "true":
        return "Low", "Demo Mode: Risk analysis placeholder."

    system = (
        "You are a legal assistant. Return JSON with keys 'risk_level' (High, Medium, Low) and 'explanation'."
    )
    user_prompt = f"Clause Type: {clause_type.value}\nText: {clause_text}"

    if _get_provider() == "gemini":
        try:
            model_instance = _gemini_model()
            full_prompt = f"{system}\n\n{user_prompt}\n\nReturn the result as a raw JSON string."
            
            response = _gemini_generate_with_retry(model_instance, full_prompt)
            if not response:
                return "Medium", "Gemini failed to generate response."
            
            # Extract JSON from potential markdown blocks
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            return data.get("risk_level", "Medium"), data.get("explanation", "No explanation.")
        except Exception as e:
            logger.error(f"Gemini Risk Error: {e}")
            if "quota" in str(e).lower() or "429" in str(e):
                return "Medium", "Gemini Quota Exceeded after retries."
            return "Medium", f"Error: {str(e)}"
    else:
        client = _openai_client()
        model_name = model or os.getenv("OPENAI_RISK_MODEL", "gpt-4.1-mini")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("risk_level", "Medium"), data.get("explanation", "No explanation.")
        except Exception as e:
            return "Medium", "OpenAI Error."

def answer_with_citations(
    question: str,
    context: Iterable[CitationChunk],
    model: str | None = None,
) -> str:
    if os.getenv("USE_MOCK_MODELS", "false").lower() == "true":
        return "Demo Mode: Citation answer placeholder."

    context_text = "\n\n".join([f"[{c.chunk_id}]: {c.content}" for c in context])
    prompt = f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer concisely with citations."

    if _get_provider() == "gemini":
        try:
            model_instance = _gemini_model()
            response = _gemini_generate_with_retry(model_instance, prompt)
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {str(e)}"
    else:
        client = _openai_client()
        model_name = model or os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1-mini")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

def chunk_ids_for_logging(chunks: Iterable[Any]) -> list[str]:
    return [c.chunk_id for c in chunks]
