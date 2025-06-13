from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from pathlib import Path
import shutil
import json
from ...core.rag import rag_system, extract_metadata_with_gemini, search_with_metadata_filter
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

@router.post("/chat")
async def chat(request: dict):
    """Stream chat response from Gemini model，结合 RAG 检索结果，并输出流式日志，SSE 支持"""
    query = request.get("query", "")
    if not query.strip():
        print("[chat] Empty query received, returning empty SSE stream.")
        return StreamingResponse((chunk for chunk in []), media_type="text/event-stream")
    # 用结构化元数据过滤检索
    results = search_with_metadata_filter(query, n_results=5)
    if not results:
        print("[chat] No results from metadata filter, fallback to vector retriever.")
        retriever = rag_system.get_retriever()
        nodes = retriever.retrieve(query)
        context = "\n\n".join([n.node.text for n in nodes])
    else:
        def format_projects(meta):
            projects_str = meta.get('项目分配', '')
            try:
                projects = json.loads(projects_str) if projects_str else []
            except Exception:
                projects = []
            if projects:
                return '\n'.join([f"  - {p['项目']}: {p['占比']}" for p in projects])
            return ''
        context = "\n\n".join([
            f"【员工】{n['metadata'].get('员工', '')}\n【时间】{n['metadata'].get('时间', '')}\n【项目】{n['metadata'].get('项目', '')}\n【项目分配】\n{format_projects(n['metadata'])}\n内容：{n['document']}"
            for n in results
        ])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(f"[chat] Full prompt sent to Gemini:\n{prompt}")
    def stream_with_log_sse():
        print("[chat] Start streaming SSE chunks...")
        chunk_count = 0
        for chunk in gemini_model.generate_stream(prompt):
            print(f"[chat] Gemini stream chunk {chunk_count}: {repr(chunk)}")
            yield f"data: {chunk}\n\n"
            chunk_count += 1
        print(f"[chat] Streaming complete, total chunks: {chunk_count}")
    return StreamingResponse(
        stream_with_log_sse(),
        media_type="text/event-stream"
    )
