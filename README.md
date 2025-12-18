# MedRAG - Medical Retrieval-Augmented Generation

An agent-orchestrated neuro-symbolic RAG system that combines explicit medical knowledge graph reasoning with neural vector retrieval to produce explainable medical question answering.

> **Educational/Research Use Only**: Not intended for actual medical diagnosis or treatment decisions.

## Overview

Multi-agent medical QA system with hybrid retrieval:
- **Knowledge Graph (Neo4j)**: Structured disease-symptom-treatment relationships
- **Vector Database (ChromaDB)**: Semantic search over 195k medicine descriptions
- **4-Agent Pipeline**: Router → KG Retrieval → Vector Retrieval → Synthesis
- **Local LLM**: Privacy-first inference with Ollama

## Architecture

```
User Query
    ↓
Agent 1: Router (intent classification, entity extraction)
    ↓
    ├─→ Agent 2: Knowledge Graph (Neo4j Cypher queries)
    └─→ Agent 3: Vector Database (ChromaDB semantic search)
    ↓
Agent 4: Synthesis (response generation + citations)
    ↓
Formatted Response
```

**Why Hybrid?**
- **KG**: Explicit relational reasoning (symptom → disease → treatment)
- **Vector DB**: Semantic search for medicine details, side effects
- **Intent-based routing**: Right retrieval strategy for each query type

## Tech Stack

**LLMs & Inference**
- **Groq API** - Fast inference for Agent 1 routing (`llama-3.3-70b-versatile`)
- [Ollama](https://ollama.ai/) - Local LLM runtime for synthesis (`llama3.2:3b`)
- [Groq](https://groq.com/) - Fast API for KG ingestion only (`llama-3.1-8b-instant`)

**RAG Orchestration**
- [LangChain](https://python.langchain.com/) - Multi-agent coordination, prompt management
- [LlamaIndex](https://www.llamaindex.ai/) - Graph/vector store abstractions

**Databases**
- [Neo4j](https://neo4j.com/) - Knowledge Graph (Cypher queries)
- [ChromaDB](https://www.trychroma.com/) - Vector store (persistent local)

**Embeddings**
- [sentence-transformers](https://www.sbert.net/) - `BAAI/bge-base-en-v1.5` (768-dim)

**Python Stack**
- `langchain-community`, `langchain-ollama`, `langchain-core`
- `llama-index-core`, `llama-index-graph-stores-neo4j`, `llama-index-vector-stores-chroma`, `llama-index-embeddings-huggingface`
- `python-dotenv`, `pydantic`, `numpy`, `tqdm`

## Installation

```bash
git clone <repo-url>
cd MedRAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Prerequisites:**
- Python 3.10+
- [Ollama](https://ollama.ai/) with `llama3.2:3b` model (for Agent 4 synthesis)
- Groq API key (for Agent 1 routing with 70B model)
- Neo4j (local or [Aura](https://neo4j.com/cloud/aura/))
- Groq API key (ingestion only)
- CUDA GPU (recommended for embedding generation)

**Setup Environment:**

Create `.env`:
```env
NEO4J_URI=neo4j+ssc://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_key
```

## Data Sources

**Disease Data** (`data/disease_data.csv`) - 110 KB, included
- Disease names, symptoms, treatments

**Medicine Data** - 338 MB, download separately
- Source: [Indian Medicine Dataset (Kaggle)](https://www.kaggle.com/datasets/mohneesh7/indian-medicine-data)
- ~195k records with compositions, side effects, interactions
- Place in `data/medicine_data.csv` after download

> **Note**: Dataset from Kaggle for demonstration purposes. Not validated for medical accuracy.

## Data Ingestion

**1. Knowledge Graph (Neo4j)**
```bash
python DB/ingestion_kg.py
```
- Processes disease CSV with Groq LLM for normalization
- Creates Disease/Symptom nodes and relationships
- ~10-20 minutes

**2. Vector Database (ChromaDB)**
```bash
python DB/ingestion_vdb.py
```
- Generates embeddings for 195k medicines
- Stores in `./DB/chroma_db/`
- ~60-90 minutes with GPU, 4-6 hours without

## Usage

```bash
python main.py
```

**Example Session:**
->In examples folder.

**Query Types:**
- Symptoms: "I have fever and cough"
- Diseases: "What is diabetes?"
- Medicines: "Side effects of aspirin"

## Project Structure

```
MedRAG/
├── agents/
│   ├── agent1_router.py          # Query classification
│   ├── agent2_kg_retrieval.py    # Neo4j queries
│   ├── agent3_vector_retrieval.py # ChromaDB search
│   └── agent4_synthesis.py       # Response generation
├── DB/
│   ├── ingestion_kg.py           # KG ingestion
│   ├── ingestion_vdb.py          # Vector DB ingestion
│   └── chroma_db/                # Vector storage (generated)
├── data/
│   ├── disease_data.csv          # Included
│   └── medicine_data.csv         # Download separately
├── examples/
├── main.py                        # Orchestrator & CLI
├── requirements.txt
├── .env                           # Create this
└── README.md
```

## Key Features

- **Intent-based routing**: Directs queries to appropriate retrieval strategy
- **Hybrid retrieval**: Combines structured (KG) and unstructured (Vector) data
- **Citation tracking**: Every fact attributed to source (KG or Vector DB)
- **Local-first**: All runtime inference via Ollama (no external API calls while running)
- **Graceful degradation**: Works with partial data availability

## Performance

- **Query latency**: 5-10 seconds (after model initialization)
- **Resource usage**: 8GB RAM, 4GB VRAM
- **Disk**: ~5GB (2GB ChromaDB + dependencies)

## Limitations

- Dataset is from Kaggle, not medically validated
- Not suitable for production medical use
- No personalization (age, allergies, interactions)
- LLM responses may vary between runs

## Documentation

- [ARCHITECTURE.md](architecture.py) - Detailed technical architecture
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Workflow explanation

## Acknowledgments

Built with: [LangChain](https://github.com/langchain-ai/langchain), [LlamaIndex](https://github.com/run-llama/llama_index), [Ollama](https://github.com/ollama/ollama), [Neo4j](https://github.com/neo4j/neo4j), [ChromaDB](https://github.com/chroma-core/chroma), [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

Dataset: [Indian Medicine Data](https://www.kaggle.com/datasets/mohneesh7/indian-medicine-data)

## License
Educational use. See individual dependencies for licenses.
