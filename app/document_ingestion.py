import os
import hashlib
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct
)
import ollama

from app.config import get_settings

settings = get_settings()


class DocumentIngestion:
    def __init__(self):
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._ensure_collection()
    
    def _ensure_collection(self):
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == settings.qdrant_collection for c in collections)
        
        if not exists:
            test_embedding = self._get_embedding("test")
            self.qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=len(test_embedding),
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {settings.qdrant_collection}")
    
    def _get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(
            model=settings.ollama_embedding_model,
            prompt=text
        )
        return response["embedding"]
    
    def _generate_doc_id(self, content: str, source: str) -> str:
        hash_input = f"{source}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def load_document(self, file_path: str) -> List[dict]:
        path = Path(file_path)
        
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif path.suffix.lower() == ".docx":
            loader = Docx2txtLoader(file_path)
        elif path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        return [
            {
                "content": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "filename": path.name,
                    **chunk.metadata
                }
            }
            for chunk in chunks
        ]
    
    def ingest_document(self, file_path: str) -> int:
        chunks = self.load_document(file_path)
        points = []
        
        for chunk in chunks:
            embedding = self._get_embedding(chunk["content"])
            doc_id = self._generate_doc_id(chunk["content"], file_path)
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "source": chunk["metadata"]["source"],
                    "filename": chunk["metadata"]["filename"],
                    "guideline_type": self._classify_guideline(file_path)
                }
            ))
        
        self.qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points
        )
        
        print(f"Ingested {len(points)} chunks from {file_path}")
        return len(points)

    def ingest_directory(self, directory: str = None) -> int:
        dir_path = Path(directory or settings.docs_directory)
        total_chunks = 0
        
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
                try:
                    chunks = self.ingest_document(str(file_path))
                    total_chunks += chunks
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        return total_chunks
    
    def _classify_guideline(self, file_path: str) -> str:
        path_lower = file_path.lower()
        
        if "aha" in path_lower or "heart" in path_lower:
            return "AHA"
        elif "ada" in path_lower or "diabetes" in path_lower:
            return "ADA"
        elif "dash" in path_lower:
            return "DASH"
        elif "cholesterol" in path_lower or "lipid" in path_lower:
            return "LIPID"
        else:
            return "GENERAL"
    
    def get_collection_stats(self) -> dict:
        info = self.qdrant.get_collection(settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count
        }
