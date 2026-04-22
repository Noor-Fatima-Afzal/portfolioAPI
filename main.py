from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import CVRAGPipeline


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources_used: int


app = FastAPI(
    title="Portfolio CV RAG API",
    version="1.0.0",
    description="RAG API that answers questions about the portfolio owner using CV context.",
)

pipeline = CVRAGPipeline()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "cv_loaded": pipeline.is_ready,
        "chunks": pipeline.chunk_count,
        "model": pipeline.model_name,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=500,
            detail="CV knowledge base is not ready. Check CV_PATH and restart the API.",
        )

    answer, source_count = pipeline.answer_question(request.question, top_k=request.top_k)
    return ChatResponse(answer=answer, sources_used=source_count)
