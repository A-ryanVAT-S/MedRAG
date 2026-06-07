# MedRAG Project Summary

I built MedRAG as an educational medical question-answering system. I designed it around a
single tool-calling agent that reasons over two grounded data sources — a Neo4j knowledge
graph of diseases and a ChromaDB vector store of ~195k medicines — and answers only from what
it retrieves.

## Why I chose this design

My earlier versions used a fixed router-then-pipeline of four agents. That made every new
query shape (drug interactions, "will X help me", reverse lookups) a new hardcoded path. I
replaced the routing and synthesis with one central agent that knows its tools and the exact
data behind each one, so it can plan and combine retrieval itself and say plainly when
something is outside the dataset.

## How it works

1. The user asks a question through one of two frontends — a CLI (`main.py`) or a Streamlit
   chat app (`streamlit/app.py`).
2. The central agent (Groq `llama-3.3-70b-versatile`, tool calling) decides which tools to
   call. It can call several in sequence, for example symptoms -> diseases -> medicines.
3. The tools run the retrieval agents and return trimmed results.
4. The agent writes the final answer from those results, citing the diseases and medicines
   it used, and noting anything the dataset does not cover.

## Frontends

- **CLI** (`main.py`) — an interactive REPL; `reset` starts a new conversation, `exit` quits.
- **Streamlit** (`streamlit/app.py`) — a chat UI that streams the agent's retrieval steps live
  (which tool it called, what it found) so the reasoning is visible while it works.

Both share the same `MedRAG` engine, so the agent behaves identically in either.

## Tools I exposed

- `find_diseases_by_symptoms(symptoms)` - candidate diseases with their symptoms, treatments,
  contagious and chronic flags. Used for diagnosis and as the first step before suggesting
  medicines.
- `get_disease_details(disease_name)` - one disease's full record.
- `get_medicine_details(medicine_name)` - one medicine: composition, price, manufacturer,
  description, side effects, drug interactions.
- `search_medicines(query)` - semantic search over medicine descriptions and side effects.

## Knowledge graph

I build the graph in `DB/ingestion_kg.py` from `data/disease_data.csv`:

1. A local Ollama model (`qwen2.5:7b-instruct`) normalizes each disease's free-text symptoms
   and treatments into atomic, lowercase entities with aliases.
2. A global entity-resolution pass merges synonymous symptoms and treatments across all
   diseases into canonical nodes that carry their aliases.
3. I write nodes and edges to Neo4j:
   - `(:Disease {disease_code, name, contagious, chronic, raw_treatments})`
   - `(:Symptom {name, aliases})`, `(:Treatment {name, aliases})`
   - `(Disease)-[:HAS_SYMPTOM]->(Symptom)`, `(Disease)-[:TREATED_BY]->(Treatment)`

For retrieval (`agents/agent2_kg_retrieval.py`) I generate candidates with an exact -> alias ->
fuzzy -> BM25 cascade, rerank them with a cross-encoder to pick seed nodes, then expand the
graph to collect connected diseases, symptoms and treatments.

## Vector store

I build the store in `DB/ingestion_vdb.py` from `data/medicine_data.csv`:

1. Clean each row into a structured record and write `DB/medicines.json` (the canonical store).
2. Create two retrieval chunks per medicine, a description chunk and a side-effects chunk.
3. Embed chunks with `BAAI/bge-base-en-v1.5` and store them in ChromaDB with metadata.

For retrieval (`agents/agent3_vector_retrieval.py`) I handle exact name lookups and salt lookups
directly from `medicines.json`, and semantic queries through ChromaDB followed by a
cross-encoder rerank. I always return the full medicine record, not just the chunk.

## Honesty

The dataset stores only what the two CSVs contain. I wrote the agent's prompt to list what each
store holds and what it does not, so for causes, prevention, dosage, prognosis, age or pregnancy
specifics, or price sorting, it states the information is not in the dataset and still answers
the parts it can.

## How I evaluate it

I built a benchmark and a scoring harness in `Evaluation/` so the numbers reflect real
retrieval rather than a lookup tautology. I generate everything from the same source CSVs the
stores are built from, so the gold is canonical.

`benchmark.py` writes labeled questions in a few shapes, each aligned to a real capability:

- **Disease (KG)** — `disease_by_symptoms` and `disease_differential` describe symptoms without
  naming the disease (semantic), and `disease_by_name` names it (known-item control).
- **Medicine semantic (vector)** — only what the vector DB actually indexes: `by_indication`
  (from the description) and `by_side_effect` (from the side-effects). I keep these out of the
  exact-lookup path on purpose.
- **Medicine known-item** — composition, price, manufacturer, interactions of a *named* medicine.
  These are exact `get_medicine_details` lookups, not semantic search.
- **Hybrid** — symptoms -> disease -> treatment (multi-hop).

Semantic questions never leak the answer name (a leak guard rejects any that do); known-item
questions name the target on purpose. Every row carries a ground-truth answer, an
`expected_tools` label, a `gold` accept-set, `difficulty` / `retrieval_type` tags, and a
`score_mode`.

`evaluation.py` runs each question and scores retrieval (`recall@k`, `MRR`, `nDCG@5`), tool
routing (`routing_recall`, agent mode), and answer quality (evidence recall, gold mention,
LLM-as-judge). I drive retrieval from the question, never the gold name. For the open-ended
medicine questions I score by **relevance** — a hit is any retrieved medicine that genuinely
covers the indication or lists the side effect, not one exact product — which is what
`search_medicines` is actually for. I slice every metric by `retrieval_type` (known_item /
semantic / multi_hop) so an exact lookup is never mistaken for semantic retrieval. Per-query
results stream to `eval_partial.jsonl` so a crash keeps what ran, and the full report —
including each answer and its retrieval trace — goes to `eval.json`.

## Limitations

- Data comes from Kaggle CSVs and I have not medically validated it.
- Not for real diagnosis or treatment decisions.
- No personalization (age, allergies, history).
- LLM responses can vary between runs.
