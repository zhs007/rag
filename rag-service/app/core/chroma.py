import chromadb
from ..config import settings
from pathlib import Path
from llama_index.embeddings.gemini import GeminiEmbedding
from chromadb import EmbeddingFunction, Documents, Embeddings
from typing import List

class ChromaGeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "models/embedding-001"):
        self.gemini_embed_model = GeminiEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # GeminiEmbedding expects a list of BaseNode or str
        # Here, 'input' is a list of strings (Documents)
        embeddings = self.gemini_embed_model.get_text_embedding_batch(input)
        return embeddings

class ChromaDBManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(Path(settings.chroma_db_path).absolute())
        )
        
        # Attempt to delete the collection if it exists to ensure fresh creation with correct dimensions
        try:
            self.client.delete_collection("rag_documents")
            print("Existing ChromaDB collection 'rag_documents' deleted.")
        except Exception as e:
            print(f"Could not delete ChromaDB collection (might not exist): {e}")

        # Explicitly specify the embedding function for the collection
        self.embedding_function = ChromaGeminiEmbeddingFunction(model_name="models/embedding-001")
        self.collection = self.client.get_or_create_collection(
            "rag_documents",
            embedding_function=self.embedding_function
        )
        print(f"ChromaDB collection 'rag_documents' initialized with {self.collection.count()} documents.")

    def add_documents(self, documents: list[str], ids: list[str], metadatas: list[dict] = None):
        """Add documents to ChromaDB collection"""
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        """Query documents from ChromaDB"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
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

chroma_manager = ChromaDBManager()
