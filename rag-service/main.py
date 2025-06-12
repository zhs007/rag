from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.rag import router as rag_router
from app.config import settings
from app.core.rag import rag_system
import uvicorn

app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可根据需要指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)

@app.on_event("startup")
async def startup_event():
    print("Ingesting documents on startup...")
    try:
        rag_system.ingest_documents()
        print("Document ingestion complete.")
    except Exception as e:
        print(f"[startup] Document ingestion failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.env == "dev"
    )
