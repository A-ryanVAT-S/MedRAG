#Vector-DB Retrieval Agent for Medicine Documents
import json
import re
import os
from pathlib import Path
from typing import Dict, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


class VectorRetrievalAgent:
    def __init__(
        self,
        chroma_path: str = None,
        collection_name: str = "medicines",
        top_k: int = 5
    ):
        #init parameters
        self.top_k = top_k
        
        # Set default path relative to project root
        if chroma_path is None:
            project_root = Path(__file__).parent.parent
            chroma_path = str(project_root / "DB" / "chroma_db")
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5"
        )

        
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            
            self.collection = self.chroma_client.get_collection(
                name=collection_name
            )
            
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            self.connected = True
            print(" [Agent 3] Connected to ChromaDB Vector Store")
            
        except Exception as e:
            print(f" [Agent 3] Failed to connect to ChromaDB: {e}")
            self.connected = False
            self.retriever = None

    def _extract_name_from_text(self, text: str) -> str:
        match = re.search(r"Medicine Name:\s*(.+)", text)
        return match.group(1).strip() if match else "Unknown Medicine"

    def _is_medically_relevant(self, text: str, query: str) -> bool:
        q = query.lower()
        t = text.lower()

        if any(k in q for k in ["pain", "fever", "antipyretic", "analgesic"]):
            if "antibiotic" in t or "antifungal" in t:
                return False

        if "influenza" in q:
            if "antifungal" in t:
                return False

        return True
    
    def retrieve_medicines(
        self,
        query: str,
        disease_context: Optional[str] = None
    ) -> Dict:
        
        #check connection
        if not self.connected:
            return self._get_fallback_response("Vector DB not connected")
        
        # Enhance query with disease context
        enhanced_query = query
        if disease_context:
            enhanced_query = f"{query} {disease_context}"
        
        print(f"\n [Agent 3] Searching vector DB: '{enhanced_query[:100]}...'")
        
        try:
            # Retrieve similar documents
            retrieved_nodes = self.retriever.retrieve(enhanced_query)
            
            medicines = []
            seen = set()

            for node in retrieved_nodes:
                if not self._is_medically_relevant(node.text, enhanced_query):
                    continue

                name = (
                    node.metadata.get("product_name")
                    or self._extract_name_from_text(node.text)
                )

                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)

                medicine_info = {
                    "name": name,
                    "summary": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                    "full_text": node.text,
                    "manufacturer": node.metadata.get("manufacturer"),
                    "category": node.metadata.get("sub_category"),
                    "similarity_score": node.score
                }
                medicines.append(medicine_info)
            
            print(f"   Retrieved {len(medicines)} medicines")
            
            return {
                "medicines": medicines,
                "source": "vector_database",
                "query": enhanced_query,
                "top_k": self.top_k
            }
            
        except Exception as e:
            print(f" Error querying vector DB: {e}")
            return self._get_fallback_response(str(e))
    
    def retrieve_by_disease(self, disease_name: str) -> Dict:
        
        # Retrieve medicines related to a specific disease
        query = f"Medicines for treating {disease_name}. Treatment options for {disease_name}."
        return self.retrieve_medicines(query, disease_context=disease_name)
    
    def retrieve_by_medicine_name(self, medicine_name: str) -> Dict:
        # Retrieve information about a specific medicine
        query = medicine_name
        return self.retrieve_medicines(query)
    
    def _get_fallback_response(self, error_msg: str) -> Dict:
        # Return empty response on failure
        return {
            "medicines": [],
            "source": "vector_database",
            "error": error_msg,
            "query": "fallback"
        }


if __name__ == "__main__":
    # Test the agent
    agent = VectorRetrievalAgent()
    
    if agent.connected:
        # Test medicine search
        print("\n" + "="*60)
        result = agent.retrieve_medicines("Pregnancy")
        print(json.dumps(result, indent=2))
        
        