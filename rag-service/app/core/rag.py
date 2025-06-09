from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PandasCSVReader
from llama_index.core.readers import SimpleDirectoryReader # Corrected import path
from .chroma import chroma_manager
from pathlib import Path
from ..config import settings
import os

os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

Settings.llm = Gemini(model="gemini-2.5-flash-preview-05-20")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

class RAGSystem:
    def __init__(self):
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_manager.collection)
        
    def ingest_documents(self):
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
            docs_to_add = [doc.text for doc in documents if doc.text.strip()] # Filter out empty documents
            ids_to_add = [doc.id_ for doc in documents if doc.text.strip()] # Filter out corresponding IDs
            print(f"Documents to add to ChromaDB (count: {len(docs_to_add)}): {docs_to_add[:100]}...") # Print first 100 chars
            chroma_manager.add_documents(docs_to_add, ids_to_add)
            print(f"Explicitly added {len(docs_to_add)} documents to ChromaDB.")

        print(f"Documents indexed. ChromaDB collection now has {chroma_manager.collection.count()} documents.")
        return index

    def get_retriever(self):
        """Create and then return a retriever from existing index"""
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        return index.as_retriever()

rag_system = RAGSystem()
