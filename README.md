# MedRAG - Medical Retrieval-Augmented Generation

I built MedRAG as an agent-orchestrated neuro-symbolic RAG system that combines an explicit
medical knowledge graph with neural vector retrieval to produce explainable medical question
answering.

> **Educational/Research Use Only**: I did not build this for actual medical diagnosis or treatment.

## Overview

I designed it around a single tool-calling agent that reasons over two grounded data sources:

- **Knowledge Graph (Neo4j)**: disease - symptom - treatment relationships
- **Vector Database (ChromaDB)**: semantic search over ~195k medicine records
- **Central Agent (Groq)**: `llama-3.3-70b-versatile` with tool calling — it plans which tools
  to use, combines results, and stays explicit about what the dataset does not contain

## Architecture

```
User Query
    |
    v
Central Agent (ReAct, tool calling)
    |
    +--> find_diseases_by_symptoms / get_disease_details   -> Neo4j KG
    +--> get_medicine_details / search_medicines           -> ChromaDB + medicines.json
    |
    v
Grounded answer (cites diseases/medicines, flags missing info)
```

I wrote the agent's prompt to describe every tool and the exact data behind it. It can chain
tools — for example symptoms -> candidate diseases -> medicines for the likely disease — and I
made it ask a clarifying question when several diseases match. See `architecture.py` for an
ASCII diagram.

**Why I went hybrid**
- The KG gives me explicit relational reasoning (symptom -> disease -> treatment)
- The vector DB gives me semantic search over medicine descriptions and side effects
- The agent decides per query which to use, instead of following a fixed pipeline

## Tech Stack

**LLMs**
- [Groq](https://groq.com/) `llama-3.3-70b-versatile` - central reasoning agent (tool calling)
- [Ollama](https://ollama.com/) `qwen2.5:7b-instruct` - local model I use for KG ingestion normalization and benchmark creation.

**Retrieval**
- [Neo4j](https://neo4j.com/) - knowledge graph (Cypher)
- [ChromaDB](https://www.trychroma.com/) - vector store (persistent, local)
- [sentence-transformers](https://www.sbert.net/) - `BAAI/bge-base-en-v1.5` embeddings (768-dim)
  and `BAAI/bge-reranker-v2-m3` cross-encoder (I share one instance across both retrieval agents)
- `rank-bm25`, `rapidfuzz` - KG candidate generation (exact/alias/fuzzy/BM25)

**Python**
- `groq`, `ollama`, `neo4j`, `chromadb`, `sentence-transformers`
- `python-dotenv`, `numpy`, `tqdm`, `pydantic`

## Installation

```bash
git clone <repo-url>
cd MedRAG
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Prerequisites**
- Python 3.10+
- Groq API key (central agent)
- Neo4j (local or [Aura](https://neo4j.com/cloud/aura/))
- Ollama with `qwen2.5:7b-instruct` pulled (KG ingestion only)
- CUDA GPU recommended for embedding and reranking

Create `.env`:
```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_key
```

## Data Sources

**Disease Data** (`data/disease_data.csv`) - included
- disease names, symptoms, treatments, contagious/chronic flags

**Medicine Data** - download separately
- Source: [Indian Medicine Dataset (Kaggle)](https://www.kaggle.com/datasets/mohneesh7/indian-medicine-data)
- ~195k records with compositions, side effects, interactions
- Place in `data/medicine_data.csv`

> The dataset is from Kaggle and I have not validated it for medical accuracy.

## Data Ingestion

**1. Knowledge Graph (Neo4j)** - needs Ollama running
```bash
ollama pull qwen2.5:7b-instruct
python DB/ingestion_kg.py
```
I normalize diseases with the local model, resolve synonymous symptoms/treatments into
canonical nodes, and write the graph. Logs and a readable report land in `DB/logs/`.

**2. Vector Database (ChromaDB)**
```bash
python DB/ingestion_vdb.py
```
I clean the CSV into `DB/medicines.json`, build description and side-effect chunks, embed
them, and store them in `DB/chroma_db/`. This takes ~2-3 hours on a GPU.

## Usage

```bash
python main.py
```
Interactive CLI. Type `reset` to clear the conversation, `exit` to quit.

**Example queries**
- Symptoms: "I have fever and cough, what could it be?"
- Disease: "Is scabies contagious?"
- Medicine: "Side effects of cetirizine"
- Medicine check: "Will cetirizine help my itching?"
- Interactions: "What does cetirizine interact with?"

I also ship a Streamlit frontend:
```bash
streamlit run streamlit/app.py
```

## Evaluation

I built a benchmark generator and a scoring harness so I can measure the system honestly
instead of trusting it by feel.

**Benchmark** (`Evaluation/benchmark.py`) — I generate labeled questions straight from the
source CSVs the stores are built from, each aligned to a real capability:

- **Disease (KG)**: symptom-described questions (semantic) and named-disease questions (known-item)
- **Medicine semantic (vector)**: only what the vector DB indexes — `by_indication` (description)
  and `by_side_effect` (side-effects)
- **Medicine known-item**: composition / price / manufacturer / interactions of a *named* medicine,
  routed to exact lookup, not semantic search
- **Hybrid**: symptoms -> disease -> treatment (multi-hop)

Semantic questions never leak the answer name (a leak guard rejects any that do); known-item
questions name the target on purpose as a labeled control. Every row carries a `ground_truth`
answer, an `expected_tools` label, a `gold` accept-set, and `difficulty` / `retrieval_type` /
`score_mode` tags.

```bash
python Evaluation/benchmark.py          # writes Evaluation/benchmark.csv
```

**Scoring** (`Evaluation/evaluation.py`) — I run every question (driven by the question, never
the gold name) and report:
- retrieval quality: `recall@1/3/5`, `MRR`, `nDCG@5`
- tool routing: `routing_recall` (did the agent call the expected tool)
- answer quality: evidence recall, gold mention, and an LLM-as-judge verdict

For the open-ended medicine questions I score by **relevance** — a hit is any retrieved medicine
that genuinely covers the indication or lists the side effect, not one exact product. I slice
all metrics by `retrieval_type` (known_item vs semantic vs multi_hop) so an exact lookup never
gets confused for real semantic retrieval. Results stream to `Evaluation/eval_partial.jsonl` as
they complete and the full report is written to `Evaluation/eval.json` (with per-query answers
and retrieval traces for debugging).

```bash
python Evaluation/evaluation.py --mode agent     # full agent (Groq); measures routing too
python Evaluation/evaluation.py --mode retrieval # retrievers only, no LLM
```

## Project Structure

```
MedRAG/
|-- agents/
|   |-- agent2_kg_retrieval.py      # Neo4j candidate generation + rerank + expansion
|   |-- agent3_vector_retrieval.py  # ChromaDB exact/salt/semantic + rerank
|   |-- agent_core.py               # central tool-calling agent
|   `-- reranker.py                 # shared cross-encoder singleton
|-- DB/
|   |-- ingestion_kg.py             # KG ingestion (Ollama + entity resolution)
|   |-- ingestion_vdb.py            # vector ingestion
|   |-- chroma_db/                  # vector storage (generated)
|   `-- medicines.json              # canonical medicine records (generated)
|-- Evaluation/
|   |-- benchmark.py                # labeled benchmark generator
|   |-- evaluation.py               # retrieval + routing + answer scoring
|   `-- benchmark.csv               # generated benchmark
|-- data/
|   |-- disease_data.csv            # included
|   `-- medicine_data.csv           # download separately
|-- streamlit/app.py                # Streamlit chat frontend
|-- main.py                         # CLI entry point
|-- architecture.py                 # ASCII architecture diagram
|-- requirements.txt
`-- .env                            # create this
```

## Limitations

- The dataset is from Kaggle and I have not medically validated it
- Not suitable for production or real medical use
- No personalization (age, allergies, patient history)
- LLM responses may vary between runs

## Acknowledgments

I built this with [Groq](https://groq.com/), [Ollama](https://ollama.com/), [Neo4j](https://neo4j.com/),
[ChromaDB](https://www.trychroma.com/), and [sentence-transformers](https://www.sbert.net/).

Dataset: [Indian Medicine Data](https://www.kaggle.com/datasets/mohneesh7/indian-medicine-data)

## License

Educational use. See individual dependencies for licenses.
