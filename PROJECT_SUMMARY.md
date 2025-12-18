# MedRAG Project Summary

## Project Overview

MedRAG (Medical Retrieval-Augmented Generation) is a sophisticated medical question-answering system that combines structured knowledge graphs with semantic vector search to provide accurate, cited medical information. The system uses a multi-agent architecture where each agent handles a specific part of the query processing pipeline.

## What Problem Does This Solve?

Traditional medical information systems suffer from:
1. **Lack of context**: Simple keyword search misses relationships between symptoms and diseases
2. **No source tracking**: Users don't know where information comes from
3. **Single retrieval method**: Either structured (database) or unstructured (text search), not both
4. **Generic responses**: No personalization based on query intent

MedRAG solves these by:
1. **Hybrid retrieval**: Combines graph relationships (symptom->disease) with semantic search (treatment options)
2. **Citation tracking**: Every fact is linked to its source (Knowledge Graph or Vector Database)
3. **Intent-based routing**: Different query types use different retrieval strategies
4. **Natural language interface**: User-friendly conversation instead of database queries

## Core Concepts

### 1. Retrieval-Augmented Generation (RAG)

Instead of having the LLM generate medical information from memory (which can hallucinate), we:
1. Retrieve relevant facts from reliable sources (databases)
2. Give these facts to the LLM as context
3. LLM synthesizes the facts into natural language

This ensures:
- Factual accuracy (grounded in data)
- Source attribution (citations)
- Up-to-date information (from databases)

### 2. Multi-Agent System

Rather than one monolithic system, we have 4 specialized agents:

**Agent 1: Router** (Groq llama-3.3-70b-versatile)
- Reads the user's question
- Uses powerful 70B model to understand intent
- Extracts ALL entities: symptoms, diseases, medicines (no hardcoded lists)
- Decides which databases to query (KG, Vector, or Both)
- Handles complex multi-intent queries intelligently

**Agent 2: Knowledge Graph Agent**
- Queries Neo4j graph database
- Finds relationships: "Which diseases have these symptoms?"
- Returns structured data: disease names, treatments, symptom matches

**Agent 3: Vector Database Agent**
- Queries ChromaDB vector store
- Semantic search: "What medicines are relevant to this query?"
- Returns: medicine descriptions, side effects, dosages

**Agent 4: Synthesis Agent**
- Takes results from Agent 2 and Agent 3
- Uses Groq API (llama-3.1-8b-instant) to combine them into natural language
- Adds citations showing where each fact came from
- Includes medical disclaimer

### 3. Hybrid Retrieval

**Why Hybrid?**
- Some knowledge is best structured (Disease HAS_SYMPTOM Fever)
- Some knowledge is best unstructured (Medicine description paragraphs)
- Best results come from using both

**Example**:
Query: "I have fever and cough"
- Graph DB: "Fever + Cough -> Influenza, Common Cold" (structured relationship)
- Vector DB: "Medicines for fever treatment: Paracetamol, Ibuprofen" (semantic similarity)
- Combined: "You may have Influenza [source: KG], try Paracetamol [source: VDB]"

## How It Works: Step-by-Step

### Phase 1: Data Preparation (One-Time Setup)

#### Knowledge Graph Creation
1. Start with disease_data.csv (diseases, symptoms, treatments)
2. Run ingestion_kg.py:
   - Reads each disease record
   - Sends to Groq LLM to normalize/structure data
   - Example: "high temperature, pyrexia, hot" -> all become "fever"
   - Creates Disease nodes in Neo4j
   - Creates Symptom nodes (deduplicated)
   - Creates relationships: (Disease)-[:HAS_SYMPTOM]->(Symptom)
3. Result: Graph with ~1000 diseases, ~500 symptoms, relationships between them

#### Vector Database Creation
1. Start with medicine_data.csv (195k medicine records)
2. Run ingestion_vdb.py:
   - Reads each medicine record
   - Creates structured text document for each
   - Loads embedding model (BAAI/bge-base-en-v1.5) on GPU
   - Converts each document to 768-dimensional vector
   - Stores vectors in ChromaDB with metadata
3. Result: 195k medicine embeddings ready for similarity search

### Phase 2: Query Processing (Runtime)

#### Step 1: User Input
User types: "I have fever and cough, what should I do?"

#### Step 2: Query Routing (Agent 1)
```
Input: "I have fever and cough, what should I do?"

Agent 1 Process:
1. Send query to Groq API (llama-3.3-70b-versatile) with system prompt
2. Powerful 70B model analyzes and returns JSON:
   {
       "intent": "symptom_to_disease",
       "entities": {
           "symptoms": ["fever", "cough"],
           "disease": null,
           "medicine": null
       },
       "use_kg": true,
       "use_vector": false
   }
3. LLM decides which data sources to use:
   - use_kg: true (need disease mapping from symptoms)
   - use_vector: false (not asking about medicines yet)

Output: Complete routing decision with entity extraction
```

#### Step 3: Knowledge Graph Retrieval (Agent 2)
```
Input: symptoms = ["fever", "cough"]

Agent 2 Process:
1. Build Cypher query:
   MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
   WHERE s.name IN ["fever", "cough"]
   WITH d, COUNT(DISTINCT s) as match_count
   ORDER BY match_count DESC
   LIMIT 5
   RETURN d.name, d.treatments, match_count

2. Execute against Neo4j
3. Parse results:
   [
       {"name": "Influenza", "treatments": "Rest, fluids, antivirals", "matches": 2},
       {"name": "Common Cold", "treatments": "Symptomatic relief", "matches": 2},
       {"name": "Pneumonia", "treatments": "Antibiotics, rest", "matches": 1}
   ]

Output: Disease information with treatments
```

#### Step 4: Vector Database Retrieval (Agent 3)
```
Input: 
- Original query: "I have fever and cough"
- KG context: "Influenza, Common Cold" (from Agent 2)

Agent 3 Process:
1. Enhance query with context:
   "fever cough treatment influenza common cold"

2. Embed query using same model:
   query_text -> embedding_model -> 768-dim vector

3. Similarity search in ChromaDB:
   - Compare query vector to all 195k medicine vectors
   - Find top 5 most similar

4. Results:
   [
       {
           "name": "Paracetamol",
           "summary": "Analgesic and antipyretic for fever and pain...",
           "similarity": 0.892(can vary)
       },
       {
           "name": "Ibuprofen", 
           "summary": "NSAID for inflammation, fever, and pain...",
           "similarity": 0.856(can vary)
       }
   ]

Output: Medicine recommendations with context
```

#### Step 5: Response Synthesis (Agent 4)
```
Input:
- User query: "I have fever and cough, what should I do?"
- KG data: Influenza, Common Cold info
- Vector data: Paracetamol, Ibuprofen info

Agent 4 Process:
1. Format both data sources into readable text
2. Send to Groq API (llama-3.1-8b-instant) with synthesis prompt:
   "Combine these facts into a helpful response for the user"
3. LLM generates natural language response:
   "Based on your symptoms of fever and cough, you may have:
    
    1. Influenza - A viral respiratory infection...
       Treatment: Rest, hydration, antivirals
       
    2. Common Cold - Milder viral infection...
       Treatment: Symptomatic relief
    
    Recommended medicines:
    - Paracetamol: Reduces fever and pain, 500-1000mg every 4-6 hours
    - Ibuprofen: Anti-inflammatory, take with food
    
    If symptoms worsen or persist, consult a healthcare provider."

4. Extract citations:
   - Influenza -> from Knowledge Graph
   - Common Cold -> from Knowledge Graph  
   - Paracetamol -> from Vector Database
   - Ibuprofen -> from Vector Database

5. Add medical disclaimer

Output: Complete formatted response
```

#### Step 6: Display
The response is formatted with:
- Main answer text
- Sources used section (KG: 2 results, Vector: 2 results)
- Detailed citations with entity names
- Medical disclaimer

## Key Technologies Explained

### 1. LlamaIndex
**What**: Framework for building RAG applications
**Why**: Simplifies integration with Neo4j and ChromaDB
**How**: Provides abstractions for vector stores, graph stores, retrievers, and query engines

### 2. LangChain
**What**: Framework for LLM application development
**Why**: Easy prompt management and LLM orchestration
**How**: PromptTemplate for structured prompts

### 3. Neo4j
**What**: Graph database for connected data
**Why**: Perfect for symptom-disease relationships (graph structure)
**How**: 
- Nodes: Disease, Symptom
- Relationships: HAS_SYMPTOM
- Query language: Cypher (similar to SQL but for graphs)

### 4. ChromaDB
**What**: Vector database for embeddings
**Why**: Fast similarity search over high-dimensional vectors
**How**: 
- Stores 768-dim vectors (embeddings)
- Uses cosine similarity to find related documents
- Persistent storage (survives restarts)

### 5. Sentence-Transformers
**What**: Library for generating text embeddings
**Why**: Converts text to numerical vectors that capture semantic meaning
**How**: 
- Model: BAAI/bge-base-en-v1.5 (768 dimensions)
- Input: "Paracetamol for fever" 
- Output: [0.23, -0.45, 0.67, ..., 0.12] (768 numbers)
- Similar texts have similar vectors

## Data Flow Deep Dive

### Knowledge Graph Structure
```
(Disease: Influenza)->treatment as attribute
    |
    +--[HAS_SYMPTOM]-->(Symptom: fever)
    |
    +--[HAS_SYMPTOM]-->(Symptom: cough)
    |
    +--[HAS_SYMPTOM]-->(Symptom: fatigue)

(Disease: Common Cold)->treatment as attribute
    |
    +--[HAS_SYMPTOM]-->(Symptom: fever)
    |
    +--[HAS_SYMPTOM]-->(Symptom: cough)

Query: Diseases with [fever, cough]
Answer: Influenza (2 matches), Common Cold (2 matches)
```

### Vector Database Structure
```
Medicine Documents (stored as embeddings):

Doc 1: "Paracetamol is an analgesic and antipyretic medication used to reduce fever and relieve mild to moderate pain..."
Embedding: [0.23, 0.45, -0.12, ..., 0.67] (768 dimensions)

Doc 2: "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever, pain, and inflammation..."
Embedding: [0.19, 0.52, -0.08, ..., 0.71] (768 dimensions)

Query: "medicine for fever"
Query Embedding: [0.22, 0.47, -0.10, ..., 0.69]

Similarity Calculation:
- cosine_similarity(query, Doc1) = 0.892
- cosine_similarity(query, Doc2) = 0.856

Results: Paracetamol (0.892), Ibuprofen (0.856)
```

## Intent-Based Routing Logic

The router determines retrieval strategy based on query intent:

### Intent: symptom_to_disease
Example: "I have fever and cough"
- Use KG: YES (map symptoms to diseases)
- Use Vector: YES (get treatment recommendations)
- Reason: Need disease identification + medicine suggestions

### Intent: disease_info
Example: "What is influenza?"
- Use KG: YES (get disease properties and symptoms)
- Use Vector: YES (get treatment medications)
- Reason: Complete disease information requires both

### Intent: medicine_info
Example: "Tell me about paracetamol"
- Use KG: NO (medicines not in graph)
- Use Vector: YES (medicine details in vector DB)
- Reason: All medicine info in vector database

### Intent: mixed
Example: "I have flu, should I take aspirin?"
- Use KG: YES (verify flu information)
- Use Vector: YES (get aspirin details)
- Reason: Involves both disease and medicine

## Why This Architecture?

### Advantages of Multi-Agent Design

1. **Separation of Concerns**
   - Each agent has one job
   - Easy to debug (which agent failed?)
   - Easy to improve (upgrade one agent without touching others)

2. **Flexibility**
   - Can disable agents (e.g., run without KG)
   - Can add new agents (e.g., drug interaction checker)
   - Can swap implementations (different vector DB)

3. **Scalability**
   - Agents can run in parallel (future optimization)
   - Each agent can scale independently
   - Can distribute across machines

4. **Testability**
   - Each agent can be tested in isolation
   - Mock data for testing
   - Clear input/output contracts

### Advantages of Hybrid Retrieval

1. **Better Coverage**
   - Structured knowledge: definitive relationships
   - Unstructured knowledge: rich descriptions
   - Together: comprehensive answers

2. **Complementary Strengths**
   - Graph DB: Fast traversals, exact relationships
   - Vector DB: Semantic similarity, fuzzy matching
   - Example: "high temperature" (fuzzy) matches "fever" (exact)

3. **Improved Accuracy**
   - Cross-validation between sources
   - More evidence = higher confidence
   - Reduces hallucination (facts from both DBs)

## Performance Characteristics

### Ingestion (One-Time)
- Knowledge Graph: ~1-2 hours for 1000 diseases (includes LLM calls)
- Vector Database: ~65 minutes for 195k medicines (GPU embedding)
- Total storage: ~2GB (ChromaDB) + ~50MB (Neo4j)

### Query Processing (Runtime)
- Agent 1 (Router): 2-3 seconds (LLM inference)
- Agent 2 (KG): 0.5-1 second (graph query)
- Agent 3 (Vector): 1-2 seconds (similarity search)
- Agent 4 (Synthesis): 3-5 seconds (LLM inference)
- Total: ~10-15 seconds per query

### Optimization Opportunities
1. Cache LLM responses for common queries
2. Pre-compute popular symptom combinations
3. Parallel agent execution (Agents 2 & 3 simultaneously)
4. Use smaller LLM for routing (faster, less accurate)
5. Increase top-k, then re-rank results

## Limitations and Considerations

### Current Limitations

1. **Data Quality**
   - Accuracy depends on source CSV quality
   - No verification of medical facts
   - May contain outdated information

2. **LLM Variability**
   - Same query can produce different responses
   - LLM may occasionally misinterpret
   - Temperature setting affects consistency

3. **Coverage Gaps**
   - Not all diseases in graph database
   - Not all medicines in vector database
   - Limited to data provided

4. **No Personalization**
   - Doesn't consider patient history
   - No age/weight/condition adjustments
   - Generic recommendations only

### Important Disclaimers

1. **Educational Purpose Only**
   - Not a substitute for medical advice
   - Should not be used for diagnosis
   - Always consult healthcare providers

2. **No Liability**
   - Information may be incomplete or incorrect
   - System may malfunction or produce errors
   - Use at your own risk

3. **Privacy**
   - Queries not stored permanently
   - Local LLM (no data sent to external APIs)
   - But conversation visible in terminal

## Extension Ideas

### Short-Term Enhancements
1. Add drug interaction checking (query for medicine combinations)
2. Implement query history and conversation context
3. Add confidence scores to responses
4. Export response as PDF with citations

### Medium-Term Features
1. Add medical images (X-rays, symptoms pictures)
2. Integrate clinical guidelines and protocols
3. Multi-language support
4. Voice input/output interface

### Long-Term Vision
1. Real-time medical literature integration
2. Patient profile management (age, conditions, allergies)
3. Dosage calculation based on patient factors
4. Integration with electronic health records (EHR)
5. Continuous learning from expert feedback

## Development Insights

### Design Decisions

**Why Neo4j over SQL?**
- Disease-symptom is naturally a graph (many-to-many relationships)
- Graph queries (Cypher) more intuitive than complex SQL joins
- Easy to add new relationship types (e.g., CAUSES, PREVENTS)

**Why ChromaDB over Elasticsearch?**
- Simpler setup (single binary, no cluster)
- Built-in embedding storage
- Python-native API
- Good performance for medium-scale datasets


**Why 4 Agents?**
- More than 4: Too complex, harder to debug
- Fewer than 4: Agents doing multiple unrelated tasks
- 4 is the sweet spot: Single responsibility, clear flow

### Code Organization

```
agents/
    Each agent is self-contained
    Can be tested independently
    Clear input/output interfaces

DB/
    Data ingestion scripts
    Separate from runtime code
    Run once, then use databases

data/
    Raw CSV files
    Source of truth for medical data
    Easy to update and re-ingest

main.py
    Orchestration only
    Thin layer connecting agents
    No business logic
```

## Conclusion

MedRAG demonstrates how modern AI techniques (RAG, multi-agent systems, hybrid retrieval) can be combined to build a practical medical information system. The architecture is modular, scalable, and transparent (every fact is cited).

Key Takeaways:
1. Hybrid retrieval beats single-method retrieval
2. Multi-agent design improves maintainability
3. Local LLMs make RAG systems privacy-friendly
4. Citation tracking builds user trust
5. Proper architecture enables future enhancements

This is a foundation that can be extended with additional agents, better models, more data, and domain-specific optimizations.
