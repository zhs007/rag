from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PandasCSVReader
from llama_index.core.readers import SimpleDirectoryReader # Corrected import path
from .chroma import chroma_manager
from pathlib import Path
from ..config import settings
import os
import re
from llama_index.llms.gemini import Gemini
import json

class RAGSystem:
    def __init__(self):
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_manager.collection)

    def _ensure_llm(self):
        from llama_index.llms.gemini import Gemini
        from llama_index.embeddings.gemini import GeminiEmbedding
        from llama_index.core import Settings
        import os
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
        Settings.llm = Gemini(model=settings.gemini_model)
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

    def ingest_documents(self):
        self._ensure_llm()
        """Load and index documents from data directory"""
        file_extractor = {
            ".csv": PandasCSVReader()
        }
        documents = SimpleDirectoryReader(
            str(Path(settings.data_dir).absolute()),
            file_extractor=file_extractor
        ).load_data()
        print(f"Loaded {len(documents)} documents from {settings.data_dir}")
        
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self.vector_store
        )
        
        # Explicitly add documents to ChromaDB if they are not already there
        if chroma_manager.collection.count() == 0:
            docs_to_add = [doc.text for doc in documents if doc.text.strip()]
            ids_to_add = [doc.id_ for doc in documents if doc.text.strip()]
            metadatas_to_add = []
            for doc in documents:
                if doc.text.strip():
                    meta = {}
                    if hasattr(doc, 'metadata') and doc.metadata:
                        meta.update(doc.metadata)
                    meta['doc_id'] = doc.id_
                    m = re.match(r"([\u4e00-\u9fa5]+) 在 ([^ ]+) 投入了 (.+)", doc.text.strip())
                    if m:
                        meta['员工'] = m.group(1)
                        meta['时间'] = m.group(2)
                        projects = parse_projects(m.group(3))
                        # 只允许基本类型，list转json字符串
                        if projects:
                            meta['项目分配'] = json.dumps(projects, ensure_ascii=False)
                            meta['项目'] = '，'.join([p['项目'] for p in projects])
                    metadatas_to_add.append(meta)
            print(f"Documents to add to ChromaDB (count: {len(docs_to_add)}): {docs_to_add[:100]}...")
            chroma_manager.add_documents(docs_to_add, ids_to_add, metadatas=metadatas_to_add)
            print(f"Explicitly added {len(docs_to_add)} documents to ChromaDB.")

        print(f"Documents indexed. ChromaDB collection now has {chroma_manager.collection.count()} documents.")
        return index

    def get_retriever(self):
        self._ensure_llm()
        """Create and then return a retriever from existing index"""
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        return index.as_retriever()

# agentB: 用 Gemini 结构化抽取用户问题中的元数据

def extract_metadata_with_gemini(question: str) -> dict:
    """
    调用 Gemini，将用户问题结构化，提取员工、时间、项目等元数据。
    返回如 {'员工': '李鸣飞', '时间': '2024年1月', '项目': '月球漫步'}
    """
    prompt = f"""
请从下面的问题中提取出员工、时间、项目（如有），以JSON格式返回。例如：{{"员工": "...", "时间": "...", "项目": "..."}}
问题：{question}
"""
    llm = Gemini(model=settings.gemini_model, api_key=settings.gemini_api_key)
    resp = llm.complete(prompt)
    # 尝试解析JSON
    try:
        meta = json.loads(resp.text.strip().split("\n")[-1])
        return meta
    except Exception:
        return {}

# agentA: 示例如何用结构化元数据过滤检索
def search_with_metadata_filter(question: str, n_results: int = 5):
    meta = extract_metadata_with_gemini(question)
    # 构造 ChromaDB 的 where 过滤条件
    where = {}
    for k in ["员工", "时间", "项目"]:
        if k in meta and meta[k]:
            where[k] = meta[k]
    # 检索
    results = chroma_manager.collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where if where else None
    )
    # 返回文档和元数据
    return [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

rag_system = RAGSystem()

def parse_projects(text: str):
    """
    解析类似“10% 在 Solar Reel，10% 在 Mystic Multiplier...”的项目分配，返回项目列表。
    返回如：[{"项目": "Solar Reel", "占比": "10%"}, ...]
    """
    # 匹配“10% 在 Solar Reel”这类片段
    pattern = r"(\d+%)[\u3000\s]*在[\u3000\s]*([\w\u4e00-\u9fa5 ]+)"
    matches = re.findall(pattern, text)
    return [{"项目": proj.strip(), "占比": percent} for percent, proj in matches]
