# Prints an ASCII overview of the MedRAG architecture.

ARCHITECTURE = r"""
                              MedRAG Architecture

                  +-----------+              +----------------------+
                  |  User CLI |              |  Streamlit frontend  |
                  |  main.py  |              |  streamlit/app.py    |
                  +-----+-----+              +-----------+----------+
                         \                              /
                          \____________  ____________/
                                       \/
                                       v
                    +--------------------------------------+
                    |        Central Agent (ReAct)         |
                    |        agents/agent_core.py          |
                    |  Groq llama-3.3-70b, tool calling    |
                    |                                      |
                    |  Knows every tool and the exact      |
                    |  data behind it. Plans which tools   |
                    |  to call, combines results, and is   |
                    |  explicit about what the data lacks. |
                    +------+----------------------+--------+
                           |                      |
              tools: KG    |                      |  tools: Vector
                           v                      v
            +---------------------------+  +---------------------------+
            |   Agent 2: KG retrieval   |  | Agent 3: Vector retrieval |
            |   exact/alias/fuzzy/BM25  |  | exact / salt / semantic   |
            |   + reranker -> expand    |  | + reranker -> full record |
            +-------------+-------------+  +-------------+-------------+
                          |                              |
                          v                              v
                 +-----------------+            +--------------------+
                 |  Neo4j (KG)     |            |  ChromaDB + JSON   |
                 |  Disease        |            |  ~195k medicines   |
                 |  Symptom        |            |  desc + side_fx    |
                 |  Treatment      |            |  chunks + records  |
                 +-----------------+            +--------------------+

Shared cross-encoder reranker (BAAI/bge-reranker-v2-m3) is loaded once and used
by both retrieval agents.

Tools available to the central agent:
  find_diseases_by_symptoms(symptoms)   -> candidate diseases + symptoms + treatments
  get_disease_details(disease_name)     -> one disease: symptoms, treatments, flags
  get_medicine_details(medicine_name)   -> one medicine: composition, side effects, interactions
  search_medicines(query)               -> semantic search over descriptions + side effects

Data the system does NOT have (the agent says so when asked): disease causes,
prevention, prognosis, medicine dosage, age/pregnancy-specific safety, price sorting.

------------------------------------------------------------------------------
Ingestion pipeline (offline, run once to build the stores)

    data/disease_data.csv                 data/medicine_data.csv
         |                                       |
         v                                       v
    DB/ingestion_kg.py                     DB/ingestion_vdb.py
      Ollama normalizes symptoms/            clean rows -> medicines.json
      treatments -> atomic entities          build desc + side-effect chunks
      global fuzzy entity resolution         embed (bge-base) -> ChromaDB
         |                                       |
         v                                       v
      Neo4j knowledge graph                  ChromaDB + medicines.json

------------------------------------------------------------------------------
Evaluation pipeline (offline)

    data/*.csv  (same sources the stores are built from)
         |
         v
    benchmark.py  -- Ollama writes labeled questions --------------------+
      semantic items hide the answer name (leak guard)                   |
      known-item items name the target (exact-lookup control)            |
      every row: ground_truth, gold, expected_tools,                     v
      difficulty, retrieval_type, score_mode                       benchmark.csv
                                                                         |
                                                                         v
    evaluation.py  -- runs each question through the real retrievers/agent
      retrieval quality : recall@1/3/5, MRR, nDCG@5
      tool routing      : routing_recall (agent mode)
      answer quality    : evidence recall, gold mention, LLM-as-judge
      relevance scoring : medicine description/side-effect questions count any
                          genuinely-valid medicine, not one exact product
      sliced by retrieval_type (known_item / semantic / multi_hop)
         |
         +--> eval_partial.jsonl  (streamed per query, crash-safe)
         `--> eval.json           (full report + per-query answers & traces)
"""


def print_architecture():
    """Print the ASCII architecture overview."""
    print(ARCHITECTURE)


if __name__ == "__main__":
    print_architecture()
