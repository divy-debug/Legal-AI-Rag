
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from embeddings import build_embeddable_items, upsert_embeddings
from legal_ai import ClauseDetector, RiskClassifier
from llm import answer_with_citations
from parser import load_document
from retrieval import hybrid_search
from utils import build_chunks

app = FastAPI(title="Privacy-First Legal AI RAG")

# Setup paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
IMAGE_DIR = BASE_DIR / "extracted_images"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/extracted_images", StaticFiles(directory="extracted_images"), name="images")
templates = Jinja2Templates(directory="templates")

# In-memory storage for the current session (for MVP simplicity)
# In a real app, this would be handled by Qdrant and a database
state = {
    "chunks": [],
    "parsed": None
}

class QuestionRequest(BaseModel):
    question: str
    collection_name: str

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    use_mock = os.getenv("USE_MOCK_MODELS", "false").lower() == "true"
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "use_mock": use_mock
    })

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form("legal_ai_gemini_v3")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Parse
        parsed = load_document(str(file_path))
        
        # 2. Detect & Classify
        detector = ClauseDetector()
        clauses = detector.detect(parsed)
        
        classifier = RiskClassifier()
        risks = classifier.classify(clauses)
        
        # 3. Chunk & Index
        chunks = build_chunks(parsed, clauses, risks)
        state["chunks"] = chunks
        state["parsed"] = parsed
        
        items = build_embeddable_items(chunks, clauses)
        upsert_embeddings(collection_name, items)
        
        # 4. Prepare images for UI
        images = []
        for el in parsed.elements:
            if el.element_type == el.element_type.IMAGE and el.metadata.get("image_path"):
                # Convert local path to URL path
                rel_path = Path(el.metadata["image_path"]).name
                images.append({
                    "url": f"/extracted_images/{rel_path}",
                    "caption": el.content
                })
        
        return {
            "status": "success",
            "clauses": [c.model_dump() for c in clauses],
            "risks": [r.model_dump() for r in risks],
            "images": images
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if not state["chunks"]:
        raise HTTPException(status_code=400, detail="No document uploaded yet")
    
    try:
        results = hybrid_search(req.collection_name, req.question, top_k=5)
        
        id_to_chunk = {c.chunk_id: c for c in state["chunks"]}
        retrieved_chunks = []
        for r in results:
            chunk = id_to_chunk.get(r.id)
            if chunk:
                retrieved_chunks.append(chunk)
        
        answer = answer_with_citations(req.question, retrieved_chunks)
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
