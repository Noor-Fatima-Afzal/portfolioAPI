from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import CVRAGPipeline


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources_used: int
    confidence: str  # "high", "medium", "low", "no_data"
    uses_cv_data: bool  # Explicit flag: is this based on CV or a refusal?


app = FastAPI(
    title="Portfolio CV RAG API",
    version="2.0.0",
    description="High-accuracy RAG API that answers questions about the portfolio owner using CV context. "
                "Prioritizes factual correctness over speculative answers.",
)

pipeline = CVRAGPipeline()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "cv_loaded": pipeline.is_ready,
        "chunks": pipeline.chunk_count,
        "model": pipeline.model_name,
        "mode": "strict_accuracy",
        "min_relevance_threshold": 0.05,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=500,
            detail="CV knowledge base is not ready. Check CV_PATH and restart the API.",
        )

    # Validate question format
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty or whitespace only.")

    answer, source_count = pipeline.answer_question(request.question, top_k=request.top_k)
    
    # Determine confidence and whether this is CV-based data
    # Refusal patterns indicate low/no confidence
    refusal_patterns = [
        "i don't have",
        "not mentioned in the cv",
        "this isn't mentioned",
        "information is not available",
        "i couldn't find",
        "no information",
        "would you like to ask",
    ]
    
    is_refusal = any(pattern in answer.lower() for pattern in refusal_patterns)
    
    if is_refusal:
        confidence = "no_data" if "i don't have" in answer.lower() else "low"
        uses_cv = False
    elif source_count >= 2:
        confidence = "high"
        uses_cv = True
    elif source_count == 1:
        confidence = "medium"
        uses_cv = True
    else:
        confidence = "low"
        uses_cv = False
    
    return ChatResponse(
        answer=answer,
        sources_used=source_count,
        confidence=confidence,
        uses_cv_data=uses_cv,
    )
