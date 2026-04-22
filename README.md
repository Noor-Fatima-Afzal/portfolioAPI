# CV RAG API (Groq)

This project exposes a FastAPI endpoint that answers questions about you from your CV using a Retrieval-Augmented Generation (RAG) pipeline and Groq.

## Endpoints

- `GET /health` - API and CV index status
- `POST /chat` - Ask questions about the candidate

Example request:

```json
{
  "question": "What are her core skills?",
  "top_k": 4
}
```

## Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start server:
   ```bash
   uvicorn main:app --reload
   ```
3. Open docs:
   - `http://127.0.0.1:8000/docs`

## Notes

- Update `CV_PATH` in `.env` if your CV filename/path changes.
- The model is set to `llama-3.3-70b-versatile` by default.
