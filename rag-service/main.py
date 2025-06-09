from fastapi import FastAPI
from app.api.v1.rag import router as rag_router
from app.config import settings
from app.core.rag import rag_system
import uvicorn

app = FastAPI(title="RAG Service")
app.include_router(rag_router)

@app.on_event("startup")
async def startup_event():
    print("Ingesting documents on startup...")
    rag_system.ingest_documents()
    print("Document ingestion complete.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.env == "dev"
    )
