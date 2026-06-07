# Shared cross-encoder reranker loaded once and reused by both retrieval agents.
import functools

SHARED_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@functools.lru_cache(maxsize=2)
def get_reranker(model_name: str = SHARED_RERANKER_MODEL):
    """Load the cross-encoder once per model name (cached for the process)."""
    import torch
    from sentence_transformers import CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [Reranker] Loading {model_name} on {device} (shared, one-time)...")
    return CrossEncoder(model_name, device=device)
