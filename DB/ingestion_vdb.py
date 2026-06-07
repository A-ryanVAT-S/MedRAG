# Vector DB ingestion: clean CSV -> medicines.json -> chunks -> embeddings -> ChromaDB.
import os
import csv
import sys
import json
import logging
import pickle
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

from datetime import datetime

# Allow very large CSV fields (medicine descriptions are long)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Logging: console (INFO) + timestamped file (DEBUG)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"vdb_ingestion_{_run_stamp}.log"

logger = logging.getLogger("vdb_ingestion")
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _console = logging.StreamHandler()
    _console.setLevel(logging.INFO)
    _console.setFormatter(_fmt)
    _file = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _file.setLevel(logging.DEBUG)
    _file.setFormatter(_fmt)
    logger.addHandler(_console)
    logger.addHandler(_file)
    logger.info(f"Logging to {LOG_FILE}")

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")

# Config
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
MEDICINES_JSON_FILE = Path(__file__).parent / "medicines.json"
EMBEDDINGS_CACHE_FILE = Path(__file__).parent / "embeddings_cache.pkl"
COLLECTION_NAME = "medicines"
BATCH_SIZE = 256
CHROMA_CHUNK_SIZE = 1000


@dataclass
class Medicine:
    """Cleaned medicine record (the canonical store)."""
    medicine_id: int
    product_name: str
    sub_category: str
    salt_composition: str
    manufacturer: str
    price: float | None
    medicine_desc: str
    side_effects: list[str] = field(default_factory=list)
    drug_interactions: list[dict] = field(default_factory=list)


def clean_price(raw: str) -> float | None:
    """Strip currency symbols and commas, e.g. 'Rs.133.93' -> 133.93."""
    if not raw:
        return None
    cleaned = re.sub(r"[^\d.]", "", raw)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_side_effects(raw: str) -> list[str]:
    """Split the comma-separated side-effects string into a list."""
    if not raw or not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def parse_drug_interactions(raw: str) -> list[dict]:
    """Zip the CSV's parallel drug/effect arrays into [{drug, effect}]."""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    drugs = data.get("drug", []) or []
    effects = data.get("effect", []) or []

    interactions = []
    for i, drug in enumerate(drugs):
        drug = (drug or "").strip()
        if not drug:
            continue
        effect = (effects[i] if i < len(effects) else "UNKNOWN") or "UNKNOWN"
        interactions.append({"drug": drug, "effect": effect.strip()})
    return interactions


def load_and_clean_csv(csv_path: str) -> list[Medicine]:
    """Read and clean the medicine CSV into Medicine records (assigns medicine_id)."""
    medicines: list[Medicine] = []
    next_id = 0
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_name = (row.get("product_name") or "").strip()
            if not product_name:
                continue
            med = Medicine(
                medicine_id=next_id,
                product_name=product_name,
                sub_category=(row.get("sub_category") or "").strip(),
                salt_composition=(row.get("salt_composition") or "").strip(),
                manufacturer=(row.get("product_manufactured") or "").strip(),
                price=clean_price(row.get("product_price", "")),
                medicine_desc=(row.get("medicine_desc") or "").strip(),
                side_effects=parse_side_effects(row.get("side_effects", "")),
                drug_interactions=parse_drug_interactions(row.get("drug_interactions", "")),
            )
            medicines.append(med)
            next_id += 1

    logger.info(f"Loaded and cleaned {len(medicines)} medicines from CSV")
    return medicines


def save_medicines_json(medicines: list[Medicine]):
    """Write the canonical record store that Agent 3 loads for fetch and exact lookup."""
    with open(MEDICINES_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump([asdict(m) for m in medicines], f, ensure_ascii=False)
    logger.info(f"Wrote {len(medicines)} records to {MEDICINES_JSON_FILE}")


def build_description_chunk(med: Medicine) -> dict:
    """Build the searchable description chunk for a medicine."""
    parts = [f"medicine name: {med.product_name}"]
    if med.sub_category:
        parts.append(f"category: {med.sub_category}")
    if med.salt_composition:
        parts.append(f"composition: {med.salt_composition}")
    if med.medicine_desc:
        parts.append(f"description:\n{med.medicine_desc}")
    text = "\n".join(parts)
    return {
        "chunk_id": f"{med.medicine_id}_desc",
        "text": text,
        "metadata": _chunk_metadata(med, "description"),
    }


def build_side_effects_chunk(med: Medicine) -> dict | None:
    """Build the side-effects chunk for a medicine, or None if it has none."""
    if not med.side_effects:
        return None
    text = (
        f"medicine: {med.product_name}\n"
        f"side effects:\n" + "\n".join(med.side_effects)
    )
    return {
        "chunk_id": f"{med.medicine_id}_side_effects",
        "text": text,
        "metadata": _chunk_metadata(med, "side_effects"),
    }


def _chunk_metadata(med: Medicine, chunk_type: str) -> dict:
    """Build Chroma metadata (values must be str/int/float/bool only)."""
    return {
        "medicine_id": med.medicine_id,
        "chunk_type": chunk_type,
        "product_name": med.product_name.lower(),
        "salt_composition": med.salt_composition.lower(),
        "sub_category": med.sub_category.lower(),
    }


def build_chunks(medicines: list[Medicine]) -> tuple[list[str], list[str], list[dict]]:
    """Build description and side-effect chunks for all medicines."""
    ids, documents, metadatas = [], [], []
    for med in medicines:
        desc = build_description_chunk(med)
        ids.append(desc["chunk_id"])
        documents.append(desc["text"])
        metadatas.append(desc["metadata"])

        se = build_side_effects_chunk(med)
        if se is not None:
            ids.append(se["chunk_id"])
            documents.append(se["text"])
            metadatas.append(se["metadata"])

    n_se = sum(1 for cid in ids if cid.endswith("_side_effects"))
    logger.info(f"Built {len(documents)} chunks from {len(medicines)} medicines "
                f"({len(medicines)} description + {n_se} side_effects)")

    # Log a few sample chunks for inspection.
    for cid, doc, meta in zip(ids[:6], documents[:6], metadatas[:6]):
        logger.debug(f"CHUNK {cid} meta={meta}\n{doc[:400]}")

    return ids, documents, metadatas


class GPUEmbedder:
    """Wraps a sentence-transformer to embed document chunks on GPU/CPU."""

    def __init__(self, model_name: str):
        """Load the embedding model onto CUDA if available, else CPU."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed texts as documents (no bge query prefix), normalized for cosine."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


def save_embeddings_cache(ids, documents, embeddings, metadatas):
    """Pickle the computed chunk embeddings so re-runs can skip embedding."""
    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump({
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }, f)
    logger.info(f"Embeddings cached to: {EMBEDDINGS_CACHE_FILE}")


def load_embeddings_cache():
    """Load the pickled embeddings cache, or None if it doesn't exist."""
    if not EMBEDDINGS_CACHE_FILE.exists():
        return None
    with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"Loaded {len(cache['documents'])} cached chunk embeddings")
    return cache


def generate_embeddings(medicines: list[Medicine]) -> tuple:
    """Build chunks, embed them, cache the result, and return all four arrays."""
    ids, documents, metadatas = build_chunks(medicines)
    embedder = GPUEmbedder(EMBEDDING_MODEL)
    logger.info(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = embedder.embed_batch(documents, batch_size=BATCH_SIZE)
    save_embeddings_cache(ids, documents, embeddings, metadatas)
    return ids, documents, embeddings, metadatas


def add_to_chromadb(ids, documents, embeddings, metadatas, start_idx: int = 0):
    """Write the embedded chunks into the Chroma collection in batches."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if start_idx == 0:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Medicine semantic chunks for MedRAG", "hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' created")
    else:
        collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"Resuming from index {start_idx}")

    embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

    logger.info("Adding chunks to ChromaDB...")
    for i in range(start_idx, len(documents), CHROMA_CHUNK_SIZE):
        end = min(i + CHROMA_CHUNK_SIZE, len(documents))
        try:
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings_list[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            logger.info(f"Added {end}/{len(documents)} chunks to ChromaDB")
        except Exception as e:
            logger.error(f"Error at index {i}: {e}")
            logger.info(f"Resume by running with --resume {i}")
            raise

    count = collection.count()
    logger.info("Vector DB ingestion complete")
    logger.info(f"  Total chunks indexed: {count}")
    logger.info(f"  Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"  ChromaDB path: {CHROMA_PERSIST_DIR}")
    logger.info(f"  Records store: {MEDICINES_JSON_FILE}")


def main():
    """Entry point: clean the CSV, embed chunks, and load them into ChromaDB."""
    csv_path = Path(__file__).parent.parent / "data" / "medicine_data.csv"
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    resume_idx = 0
    if len(sys.argv) > 2 and sys.argv[1] == "--resume":
        resume_idx = int(sys.argv[2])

    cache = load_embeddings_cache()
    if cache and "--regenerate" not in sys.argv:
        logger.info("Using cached chunk embeddings (use --regenerate to recompute)")
        ids = cache["ids"]
        documents = cache["documents"]
        embeddings = cache["embeddings"]
        metadatas = cache["metadatas"]
    else:
        medicines = load_and_clean_csv(str(csv_path))
        save_medicines_json(medicines)
        ids, documents, embeddings, metadatas = generate_embeddings(medicines)

    add_to_chromadb(ids, documents, embeddings, metadatas, start_idx=resume_idx)


if __name__ == "__main__":
    main()
