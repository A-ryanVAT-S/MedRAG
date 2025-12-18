# Vector DB Ingestion - Medicine embeddings using sentence-transformers + ChromaDB
import os
import csv
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")

# Config
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
EMBEDDINGS_CACHE_FILE = Path(__file__).parent / "embeddings_cache.pkl"
COLLECTION_NAME = "medicines"
BATCH_SIZE = 256
CHROMA_CHUNK_SIZE = 1000  


@dataclass
class Medicine:
    # Medicine record from CSV
    product_name: str
    sub_category: str
    salt_composition: str
    manufacturer: str
    description: str
    side_effects: str
    drug_interactions: str


def parse_drug_interactions(raw: str) -> str:
    # Parse JSON drug interactions into readable text
    if not raw or raw.strip() == "":
        return "No known drug interactions."
    
    try:
        data = json.loads(raw)
        drugs = data.get("drug", [])
        effects = data.get("effect", [])
        
        if not drugs:
            return "No known drug interactions."
        
        interactions = []
        for i, drug in enumerate(drugs):
            effect = effects[i] if i < len(effects) else "UNKNOWN"
            interactions.append(f"{drug} ({effect})")
        
        return "Interacts with: " + ", ".join(interactions)
    except:
        return raw.strip() if raw else "No known drug interactions."


def create_document(med: Medicine) -> str:
    # Create structured document for embedding
    doc = f"""Medicine Name: {med.product_name}

Category:
{med.sub_category}

Composition:
{med.salt_composition}

Description:
{med.description}

Common Side Effects:
{med.side_effects}

Drug Interactions:
{med.drug_interactions}

Manufacturer:
{med.manufacturer}"""
    
    return doc.strip()


class GPUEmbedder:
    # Direct GPU embedding using sentence-transformers
    
    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device="cuda")
        logger.info(f"Model loaded on GPU. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # Batch embed on GPU
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings


def load_csv(csv_path: str) -> list[Medicine]:
    # Load medicine records from CSV
    medicines = []
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            med = Medicine(
                product_name=row.get("product_name", "").strip(),
                sub_category=row.get("sub_category", "").strip(),
                salt_composition=row.get("salt_composition", "").strip(),
                manufacturer=row.get("product_manufactured", "").strip(),
                description=row.get("medicine_desc", "").strip(),
                side_effects=row.get("side_effects", "").strip(),
                drug_interactions=parse_drug_interactions(row.get("drug_interactions", ""))
            )
            if med.product_name:
                medicines.append(med)
    
    logger.info(f"Loaded {len(medicines)} medicines from CSV")
    return medicines


def save_embeddings_cache(documents: list[str], embeddings: np.ndarray, metadatas: list[dict]):
    # Save embeddings to disk
    cache_data = {
        "documents": documents,
        "embeddings": embeddings,
        "metadatas": metadatas
    }
    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    logger.info(f"Embeddings cached to: {EMBEDDINGS_CACHE_FILE}")


def load_embeddings_cache():
    # Load embeddings from disk
    if not EMBEDDINGS_CACHE_FILE.exists():
        return None
    
    with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)
    logger.info(f"Loaded {len(cache_data['documents'])} embeddings from cache")
    return cache_data


def generate_embeddings(medicines: list[Medicine]) -> tuple:
    # Generate embeddings and cache them
    embedder = GPUEmbedder(EMBEDDING_MODEL)
    
    logger.info("Creating documents...")
    documents = [create_document(med) for med in medicines]
    
    logger.info(f"Generating embeddings for {len(documents)} documents...")
    embeddings = embedder.embed_batch(documents, batch_size=BATCH_SIZE)
    
    metadatas = [
        {
            "product_name": med.product_name.lower(),
            "sub_category": med.sub_category.lower(),
            "manufacturer": med.manufacturer.lower()
        }
        for med in medicines
    ]
    
    # Save to disk immediately
    save_embeddings_cache(documents, embeddings, metadatas)
    
    return documents, embeddings, metadatas


def add_to_chromadb(documents: list[str], embeddings: np.ndarray, metadatas: list[dict], start_idx: int = 0):
    # Add embeddings to ChromaDB with resume support
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    # Get or create collection
    if start_idx == 0:
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Medicine embeddings for MedRAG"}
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' created")
    else:
        collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"Resuming from index {start_idx}")
    
    # Convert numpy to list for ChromaDB
    embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    
    # Generate IDs
    ids = [f"med_{i}" for i in range(len(documents))]
    
    # Add chunks
    logger.info("Adding to ChromaDB...")
    for i in range(start_idx, len(documents), CHROMA_CHUNK_SIZE):
        end = min(i + CHROMA_CHUNK_SIZE, len(documents))
        try:
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings_list[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
            logger.info(f"Added {end}/{len(documents)} to ChromaDB")
        except Exception as e:
            logger.error(f"Error at index {i}: {e}")
            logger.info(f"Resume from index {i} by running with --resume {i}")
            raise
    
    # Print stats
    count = collection.count()
    logger.info("=" * 50)
    logger.info("Vector DB Ingestion Complete!")
    logger.info(f"  Total medicines indexed: {count}")
    logger.info(f"  Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"  ChromaDB path: {CHROMA_PERSIST_DIR}")
    logger.info(f"  Collection: {COLLECTION_NAME}")
    logger.info("=" * 50)


def main():
    import sys
    
    csv_path = Path(__file__).parent.parent / "data" / "medicine_data.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Check for resume argument
    resume_idx = 0
    if len(sys.argv) > 2 and sys.argv[1] == "--resume":
        resume_idx = int(sys.argv[2])
    
    # Check for cached embeddings
    cache = load_embeddings_cache()
    
    if cache and "--regenerate" not in sys.argv:
        logger.info("Using cached embeddings (use --regenerate to recompute)")
        documents = cache["documents"]
        embeddings = cache["embeddings"]
        metadatas = cache["metadatas"]
    else:
        # Generate fresh
        medicines = load_csv(str(csv_path))
        documents, embeddings, metadatas = generate_embeddings(medicines)
    
    # Add to ChromaDB
    add_to_chromadb(documents, embeddings, metadatas, start_idx=resume_idx)


if __name__ == "__main__":
    main()
