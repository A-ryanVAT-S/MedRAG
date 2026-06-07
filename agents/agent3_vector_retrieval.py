# Medicine retrieval with three paths: exact name, salt lookup, and semantic search.
# Chunks live in Chroma; full records are fetched from medicines.json by medicine_id.
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

try:
    from .reranker import get_reranker, SHARED_RERANKER_MODEL
except ImportError:
    from reranker import get_reranker, SHARED_RERANKER_MODEL

# bge-base-en-v1.5 expects this instruction on the QUERY side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


class VectorRetrievalAgent:
    """Retrieves medicines by exact name, salt lookup, or Chroma semantic search."""

    def __init__(
        self,
        chroma_path: str = None,
        medicines_json: str = None,
        collection_name: str = "medicines",
        top_k: int = 5,
        candidate_pool: int = 60,
    ):
        """Load medicine records and indexes, and connect to the Chroma collection."""
        self.top_k = top_k
        self.candidate_pool = candidate_pool

        project_root = Path(__file__).parent.parent
        if chroma_path is None:
            chroma_path = str(project_root / "DB" / "chroma_db")
        if medicines_json is None:
            medicines_json = str(project_root / "DB" / "medicines.json")

        # Heavy models load lazily, only for the semantic path.
        self._embedder = None
        self._reranker = None

        # Canonical record store + in-memory indexes for exact/salt lookups.
        self.records_by_id: Dict[int, dict] = {}
        self.name_index: Dict[str, int] = {}        # normalized product_name -> medicine_id
        self.salt_index: Dict[str, List[int]] = {}  # normalized salt -> [medicine_id]
        self._load_records(medicines_json)

        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.chroma_client.get_collection(name=collection_name)
            self.connected = True
            print(f" [Agent 3] Connected to ChromaDB ({self.collection.count()} chunks)")
        except Exception as e:
            print(f" [Agent 3] Failed to connect to ChromaDB: {e}")
            self.connected = False
            self.collection = None

    def _load_records(self, path: str):
        """Load medicine records and build the name and salt indexes."""
        p = Path(path)
        if not p.exists():
            print(f" [Agent 3] medicines.json not found at {path} (run ingestion_vdb.py)")
            return
        with open(p, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            mid = rec["medicine_id"]
            self.records_by_id[mid] = rec
            self.name_index[rec["product_name"].lower().strip()] = mid
            salt = (rec.get("salt_composition") or "").lower().strip()
            if salt:
                self.salt_index.setdefault(salt, []).append(mid)
        print(f" [Agent 3] Loaded {len(self.records_by_id)} medicine records")

    def _get_embedder(self):
        """Lazily load the sentence-transformer query embedder."""
        if self._embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f" [Agent 3] Loading embedder ({device})...")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        return self._embedder

    def _get_reranker(self):
        """Return the shared cross-encoder reranker singleton (same as Agent 2)."""
        return get_reranker(SHARED_RERANKER_MODEL)

    def retrieve_medicines(self, query: str, disease_context: Optional[str] = None) -> Dict:
        """Retrieve medicines by salt lookup, falling back to semantic search."""
        if not self.records_by_id:
            return self._fallback("No medicine records loaded")

        salt_ids = self._match_salt(query)
        if salt_ids:
            return self._build_response(salt_ids, query, match_type="salt_lookup")

        enhanced = f"{query} {disease_context}".strip() if disease_context else query
        return self._semantic_retrieve(enhanced)

    def retrieve_by_medicine_name(self, medicine_name: str) -> Dict:
        """Exact product-name lookup, falling back to semantic search on a miss."""
        if not self.records_by_id:
            return self._fallback("No medicine records loaded")

        ids = self._match_name(medicine_name)
        if ids:
            return self._build_response(ids, medicine_name, match_type="exact_medicine")

        print(f" [Agent 3] No exact name match for '{medicine_name}', using semantic search")
        return self._semantic_retrieve(medicine_name)

    def retrieve_by_disease(self, disease_name: str) -> Dict:
        """Find medicines that treat a disease via a semantic query."""
        query = f"Medicines for treating {disease_name}. Treatment options for {disease_name}."
        return self.retrieve_medicines(query, disease_context=disease_name)

    def _match_name(self, name: str) -> List[int]:
        """Match a product name exactly, then by substring, returning medicine ids."""
        key = name.lower().strip()
        if not key:
            return []
        if key in self.name_index:
            return [self.name_index[key]]
        hits = [mid for n, mid in self.name_index.items() if key in n]
        return hits[: self.top_k]

    def _match_salt(self, query: str) -> List[int]:
        """Match the query to a salt composition, returning medicine ids."""
        q = query.lower()
        if q.strip() in self.salt_index:
            return self.salt_index[q.strip()][: self.top_k]
        # Salt name appears as a phrase inside the query.
        for salt, ids in self.salt_index.items():
            if len(salt) >= 5 and salt in q:
                return ids[: self.top_k]
        return []

    def _semantic_retrieve(self, query: str) -> Dict:
        """Embed the query, search Chroma, and cross-encoder rerank the results."""
        if not self.connected:
            return self._fallback("Vector DB not connected")

        print(f"\n [Agent 3] Semantic search: '{query[:100]}'")
        try:
            embedder = self._get_embedder()
            q_emb = embedder.encode(
                QUERY_INSTRUCTION + query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=self.candidate_pool,
            )

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            if not docs:
                return self._build_response([], query, match_type="semantic")

            # Cross-encoder rerank on (query, chunk_text) pairs.
            reranker = self._get_reranker()
            scores = reranker.predict([(query, d) for d in docs])

            ranked = sorted(zip(scores, metas), key=lambda x: x[0], reverse=True)

            # Dedup by product name (the dataset has duplicate records under different
            # ids) so top_k is top_k DISTINCT medicines, not one product repeated.
            ordered_ids: List[int] = []
            score_by_id: Dict[int, float] = {}
            seen_names: set = set()
            for score, meta in ranked:
                mid = meta.get("medicine_id")
                pname = meta.get("product_name") or str(mid)
                if mid is None or pname in seen_names:
                    continue
                seen_names.add(pname)
                score_by_id[mid] = float(score)
                ordered_ids.append(mid)
                if len(ordered_ids) >= self.top_k:
                    break

            return self._build_response(
                ordered_ids, query, match_type="semantic", scores=score_by_id
            )
        except Exception as e:
            print(f" [Agent 3] Semantic retrieval error: {e}")
            return self._fallback(str(e))

    def _build_response(
        self,
        medicine_ids: List[int],
        query: str,
        match_type: str,
        scores: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """Assemble full medicine records for the given ids into a response dict."""
        medicines = []
        for mid in medicine_ids:
            rec = self.records_by_id.get(mid)
            if not rec:
                continue
            medicines.append({
                "medicine_id": mid,
                "name": rec["product_name"],
                "description": rec.get("medicine_desc", ""),
                "salt_composition": rec.get("salt_composition", ""),
                "sub_category": rec.get("sub_category", ""),
                "manufacturer": rec.get("manufacturer", ""),
                "price": rec.get("price"),
                "side_effects": rec.get("side_effects", []),
                "drug_interactions": rec.get("drug_interactions", []),
                "similarity_score": (scores or {}).get(mid),
                "match_type": match_type,
            })

        print(f"   Retrieved {len(medicines)} medicines via {match_type}")
        return {
            "medicines": medicines,
            "source": "vector_database",
            "query": query,
            "query_type": match_type,
            "top_k": self.top_k,
        }

    def _fallback(self, error_msg: str) -> Dict:
        """Return an empty result carrying an error message."""
        return {
            "medicines": [],
            "source": "vector_database",
            "error": error_msg,
            "query": "fallback",
        }


if __name__ == "__main__":
    agent = VectorRetrievalAgent()
    print(json.dumps(agent.retrieve_medicines("which medicines cause swelling?"), indent=2, ensure_ascii=False)[:1500])
    print(json.dumps(agent.retrieve_by_medicine_name("Human Insulatard 40IU/ml Suspension for Injection"), indent=2, ensure_ascii=False)[:1500])
