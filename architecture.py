"""
System Architecture Visualization
ASCII art diagram of the MedRAG multi-agent system.
"""

ARCHITECTURE = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MedRAG System Architecture                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

                                  ┌─────────────┐
                                  │  USER CLI   │
                                  │   main.py   │
                                  └──────┬──────┘
                                         │
                                    User Query
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AGENT 1: Query Router                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Analyzes user intent                                              │    │
│  │  • Extracts entities (symptoms, disease, medicine)                   │    │
│  │  • Determines retrieval strategy                                     │    │
│  │  • LLM: Groq (llama-3.3-70b-versatile)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  Output: {                                                                   │
│    "intent": "symptom_to_disease | disease_info | medicine_info....",   │
│    "entities": {"symptoms": [...], "disease": "...", "medicine": "..."}     │
│  }                                                                            │
└──────────────────────────┬───────────────────┬────────────────────────────────┘
                           │                   │
                     Use KG?│                   │Use Vector?
                           ▼                   ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │  AGENT 2: KG Retrieval   │  │ AGENT 3: Vector Retrieval│
        └──────────────────────────┘  └──────────────────────────┘
                     │                           │
                     ▼                           ▼
┌─────────────────────────────────┐  ┌──────────────────────────────────┐
│     Neo4j Knowledge Graph       │  │       ChromaDB Vector Store       │
│  ┌───────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │                           │  │  │  │                            │  │
│  │    Disease ──┐            │  │  │  │    Medicine Documents      │  │
│  │       │      │            │  │  │  │    • Descriptions          │  │
│  │       │  HAS_SYMPTOM      │  │  │  │    • Side effects          │  │
│  │       │      │            │  │  │  │    • Dosage info           │  │
│  │       ▼      ▼            │  │  │  │    • Interactions          │  │
│  │   Symptom  Symptom        │  │  │  │                            │  │
│  │                           │  │  │  │    Embeddings + Similarity │  │
│  │   Treatments              │  │  │  │    Search (Semantic)       │  │
│  │   Descriptions            │  │  │  │                            │  │
│  └───────────────────────────┘  │  │  └────────────────────────────┘  │
│                                 │  │                                  │
│  Query: Cypher (Graph Pattern)  │  │  Query: Vector Similarity Search │
└──────────────┬──────────────────┘  └─────────────┬────────────────────┘
               │                                   │
               │ Structured Facts                  │ • Semantic Results
               │ • Disease properties              │ • Top-K medicines
               │ • Symptom mappings                │ • Similarity scores
               │ • Treatment info                  │ • Full text 
               │                                   │
               └──────────────┬────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT 4: Response Synthesis                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Combines KG structured facts + Vector semantic results           │    │
│  │  • Generates coherent natural language response                     │    │
│  │  • Adds citations for each fact (source tracking)                   │    │
│  │  • Includes medical safety disclaimer                               │    │
│  │  • LLM: Ollama (llama3.2:3b)                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: {                                                                  │
│    "answer": "Natural language response with [citations]",                  │
│    "citations": [{source, type, entity}...],                                │
│    "sources_used": {kg: bool, vector: bool},                                │
│    "disclaimer": "Medical safety warning"                                   │
│  }                                                                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  FORMATTED RESPONSE │
                    │  ┌───────────────┐  │
                    │  │ -> Answer     │  │
                    │  │ -> Sources    │  │
                    │  │ -> Citations  │  │
                    │  │ -> Disclaimer │  │
                    │  └───────────────┘  │
                    └─────────────────────┘
                               │
                               ▼
                         Display to User


╔═══════════════════════════════════════════════════════════════════════════════╗
║                              Data Flow Example                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Query: "I have fever and cough, what could it be?"
   │
   ▼
Agent 1: {intent: "symptom_to_disease", entities: {symptoms: ["fever", "cough"]}}
   │
   ├─────► Agent 2 (KG): Query symptoms → Find diseases
   │          Returns: [Influenza, Common Cold] with treatments
   │
   └─────► Agent 3 (Vector): Search "fever cough treatment"
              Returns: [Paracetamol, Ibuprofen] with details
   │
   ▼
Agent 4: Synthesize both sources
   Output: "Based on symptoms [KG: diseases], you may have X.
            Consider [VDB: medicines] for relief. [Disclaimer]"
   │
   ▼
Display: Formatted response with citations


╔═══════════════════════════════════════════════════════════════════════════════╗
║                           Technology Stack                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────┬──────────────────┬────────────────────────────────────────┐
│    Component     │    Technology    │            Purpose                     │
├──────────────────┼──────────────────┼────────────────────────────────────────┤
│ Orchestration    │ LangChain        │ Multi-agent workflow management        │
│ LLM              │ Ollama           │ Local inference (Llama 3.2 3B)         │
│ KG Creation      | Groq             | Using CSV+ingenstion Pipeline          │
| VectorDB Creation| BAAI/base-model  | Embedding generation for documents(csv)│
| KG Storage       │ Neo4j            │ Disease-symptom graph relationships    │
│ Vector Storage   │ ChromaDB         │ Medicine embeddings & similarity       │
│ KG Interface     │ LlamaIndex       │ Knowledge graph querying               │
│ Vector Interface │ LlamaIndex       │ Vector store retrieval                 │
│ CLI              │ Python argparse  │ Interactive & single-query modes       │
└──────────────────┴──────────────────┴────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                      Agent Communication Protocol                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  Agent 1 → Agent 2/3                                                        │
│  ─────────────────────                                                      │
│  routing_data = {                                                           │
│    "intent": str,                                                           │
│    "entities": {                                                            │
│      "symptoms": List[str],                                                 │
│      "disease": Optional[str],                                              │
│      "medicine": Optional[str]                                              │
│    }                                                                        │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Agent 2 → Agent 4                                                          │
│  ─────────────────────                                                      │
│  kg_data = {                                                                │
│    "diseases": List[{                                                       │
│      "name": str,                                                           │
│      "treatments": str,                                                     │
│      "description": str,                                                    │
│      "symptoms": List[str]                                                  │
│    }],                                                                      │
│    "source": "knowledge_graph",                                             │
│    "query_type": str                                                        │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Agent 3 → Agent 4                                                          │
│  ─────────────────────                                                      │
│  vector_data = {                                                            │
│    "medicines": List[{                                                      │
│      "name": str,                                                           │
│      "summary": str,                                                        │                                                       │
│      "similarity_score": float                                              │
│    }],                                                                      │
│    "source": "vector_database",                                             │
│    "query": str                                                             │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Agent 4 → User                                                             │
│  ─────────────────                                                          │
│  response = {                                                               │
│    "answer": str,                    # Natural language response            │
│    "citations": List[Dict],          # Source references                    │
│    "sources_used": Dict,             # Which systems were queried           │
│    "disclaimer": str,                # Medical safety warning               │
│    "routing": Dict,                  # Original routing data                │
│    "retrieval_strategy": Dict        # What was retrieved                   │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                          Execution Flow Diagram                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

START
  │
  ├─> User enters query in CLI (main.py)
  │
  ├─> MedRAGOrchestrator.process_query(query)
  │     │
  │     ├─> Agent 1: router.route_query()
  │     │     │
  │     │     ├─> LLM analyzes query
  │     │     └─> Returns: routing_data
  │     │
  │     ├─> Orchestrator checks routing decision
  │     │     │
  │     │     ├─> if should_use_kg():
  │     │     │     └─> Agent 2: kg_agent.retrieve()
  │     │     │           │
  │     │     │           ├─> Query Neo4j with Cypher
  │     │     │           └─> Returns: kg_data
  │     │     │
  │     │     └─> if should_use_vector():
  │     │           └─> Agent 3: vector_agent.retrieve()
  │     │                 │
  │     │                 ├─> Similarity search in ChromaDB
  │     │                 └─> Returns: vector_data
  │     │
  │     └─> Agent 4: synthesis_agent.synthesize()
  │           │
  │           ├─> Format KG and Vector data
  │           ├─> LLM generates synthesis
  │           └─> Returns: response (with citations)
  │
  └─> Display formatted response to user
  │
END


"""

def print_architecture():
    """Print the architecture diagram."""
    print(ARCHITECTURE)


if __name__ == "__main__":
    print_architecture()
