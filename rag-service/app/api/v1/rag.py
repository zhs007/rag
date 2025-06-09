from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from pathlib import Path
import shutil
from ...core.rag import rag_system
from ...core.gemini import gemini_model
from ...config import settings

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

@router.post("/ingest")
async def ingest_documents(file: UploadFile = File(...)):
    """Upload and process document for RAG"""
    file_path = Path(settings.data_dir) / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    rag_system.ingest_documents()
    return {"status": "success", "filename": file.filename}

@router.get("/search")
async def search(query: str):
    """Search documents using RAG"""
    retriever = rag_system.get_retriever()
    results = retriever.retrieve(query)
    
    print(f"Retrieved {len(results)} documents.")
    
    context = "\n\n".join([n.node.text for n in results])
    print(f"Context:\n{context}")
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(f"Full prompt sent to Gemini:\n{prompt}")
    
    return StreamingResponse(
        gemini_model.generate_stream(prompt),
        media_type="text/plain"
    )
