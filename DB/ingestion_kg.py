# KG Ingestion Pipeline - Builds Disease-Symptom-Treatment graph using Groq 
import os
import csv
import json
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from neo4j import GraphDatabase
from groq import Groq

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Rate limiting config
BATCH_SIZE = 1 
REQUEST_DELAY = 2.5
MAX_RETRIES = 2
INITIAL_BACKOFF = 2.0  


@dataclass
class StructuredDisease:
    # Structured disease after LLM processing
    name: str
    disease_code: str
    chronic: bool
    contagious: bool
    symptoms: list[str] = field(default_factory=list)
    treatments: str = ""  


class LLMStructurer:
    # LLM-based entity normalization using Groq 
    
    SYSTEM_PROMPT = """You are a medical knowledge graph structuring assistant. 
Your task is to normalize and structure medical entity names for consistency in a knowledge graph.

Rules:
1. Normalize entity names to proper medical terminology
2. Remove duplicates and merge similar concepts
3. Use lowercase for all entity names (disease, symptoms)
4. Split compound symptoms into individual entities
5. Remove any empty or null values
6. Trim whitespace from all names
7. Keep treatments as a single descriptive string (do not split)
8. Output ONLY valid JSON, no explanations

Output JSON schema:
{
    "disease_name": "normalized disease name in lowercase",
    "symptoms": ["symptom 1", "symptom 2"],
    "treatments": "full treatment description as string"
}"""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.cache: dict[str, dict] = {}
        self.cache_file = Path(__file__).parent / ".llm_cache.json"
        self._load_cache()
    
    def _load_cache(self) -> None:
        # Load cached LLM responses from disk
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached disease records")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        # Persist cache to disk
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, disease_name: str, symptoms: str, treatments: str) -> str:
        # Generate deterministic cache key
        content = f"{disease_name}|{symptoms}|{treatments}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def structure_disease(
        self,
        disease_name: str,
        symptoms_raw: str,
        treatments_raw: str,
        retry_count: int = 0
    ) -> Optional[dict]:
        # Structure disease record using LLM with exponential backoff
        cache_key = self._get_cache_key(disease_name, symptoms_raw, treatments_raw)
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit for: {disease_name}")
            return self.cache[cache_key]
        
        user_prompt = f"""Structure the following medical record into the JSON schema:

            Disease Name: {disease_name}
            Symptoms: {symptoms_raw}
            Treatments: {treatments_raw}

            Return ONLY the JSON object, no additional text."""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=1024
                )
            )
            
            result_text = response.choices[0].message.content
            
            # Handle empty response
            if not result_text or not result_text.strip():
                logger.error(f"Empty response for {disease_name}")
                return None
            
            # Strip markdown code blocks if present (```json ... ```)
            result_text = result_text.strip()
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                # Remove first line (```json) and last line (```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                result_text = "\n".join(lines)
            
            result = json.loads(result_text)
            
            self.cache[cache_key] = result
            self._save_cache()
            
            logger.debug(f"Structured: {disease_name}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {disease_name}: {e}")
            logger.debug(f"Raw response was: {result_text[:200] if 'result_text' in dir() else 'N/A'}")
            # Retry once on JSON parse error (LLM may have had a hiccup)
            if retry_count < 1:
                await asyncio.sleep(1)
                return await self.structure_disease(
                    disease_name, symptoms_raw, treatments_raw, retry_count + 1
                )
            return None
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle rate limiting (429) with exponential backoff
            if "429" in error_msg or "rate_limit" in error_msg:
                if retry_count < MAX_RETRIES:
                    backoff = INITIAL_BACKOFF * (2 ** retry_count)
                    logger.warning(f"Rate limited. Retrying {disease_name} in {backoff}s...")
                    await asyncio.sleep(backoff)
                    return await self.structure_disease(
                        disease_name, symptoms_raw, treatments_raw, retry_count + 1
                    )
                else:
                    logger.error(f"Max retries exceeded for {disease_name}")
                    return None
            else:
                logger.error(f"Error processing {disease_name}: {e}")
                return None


class Neo4jKnowledgeGraph:
    # Neo4j graph operations - node creation, de-duplication, relationships
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        self.driver.close()
    
    def create_constraints(self) -> None:
        # Only disease_code needs to be unique
        with self.driver.session() as session:
            try:
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS 
                    FOR (n:Disease) 
                    REQUIRE n.disease_code IS UNIQUE
                """)
                logger.info("Constraint created: Disease.disease_code")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {e}")
    
    def clear_graph(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared")
    
    def insert_disease(self, disease: StructuredDisease) -> None:
        # Insert disease with symptoms, treatments as attribute using MERGE for de-duplication
        with self.driver.session() as session:
            # Create Disease node with treatments as attribute
            session.run("""
                MERGE (d:Disease {disease_code: $code})
                SET d.name = $name,
                    d.chronic = $chronic,
                    d.contagious = $contagious,
                    d.treatments = $treatments
            """, {
                "code": disease.disease_code.strip().lower(),
                "name": disease.name.strip().lower(),
                "chronic": disease.chronic,
                "contagious": disease.contagious,
                "treatments": disease.treatments.strip().lower()
            })
            
            # Create individual Symptom nodes and HAS_SYMPTOM relationships
            for symptom in disease.symptoms:
                if symptom and symptom.strip():
                    session.run("""
                        MERGE (s:Symptom {name: $symptom_name})
                        WITH s
                        MATCH (d:Disease {disease_code: $code})
                        MERGE (d)-[:HAS_SYMPTOM]->(s)
                    """, {
                        "symptom_name": symptom.strip().lower(),
                        "code": disease.disease_code.strip().lower()
                    })
            
            logger.debug(f"Inserted disease: {disease.name}")
    
    def get_statistics(self) -> dict:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Disease) WITH count(d) as diseases
                MATCH (s:Symptom) WITH diseases, count(s) as symptoms
                MATCH ()-[r:HAS_SYMPTOM]->() 
                RETURN diseases, symptoms, count(r) as has_symptom_rels
            """)
            record = result.single()
            return {
                "diseases": record["diseases"],
                "symptoms": record["symptoms"],
                "has_symptom_relationships": record["has_symptom_rels"]
            }


class KGIngestionPipeline:
    # Main pipeline orchestrator for KG construction
    
    def __init__(self):
        self.structurer = LLMStructurer(GROQ_API_KEY)
        self.graph = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    def load_csv(self, csv_path: str) -> list[dict]:
        records = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        logger.info(f"Loaded {len(records)} records from CSV")
        return records
    
    async def process_batch(self, records: list[dict]) -> list[StructuredDisease]:
        # Process batch of records concurrently
        tasks = []
        for record in records:
            task = self.structurer.structure_disease(
                disease_name=record.get("Name", ""),
                symptoms_raw=record.get("Symptoms", ""),
                treatments_raw=record.get("Treatments", "")
            )
            tasks.append((record, task))
        
        results = []
        for record, task in tasks:
            structured = await task
            if structured:
                disease = StructuredDisease(
                    name=structured.get("disease_name", record.get("Name", "")),
                    disease_code=record.get("Disease_Code", ""),
                    chronic=record.get("Chronic", "False").lower() == "true",
                    contagious=record.get("Contagious", "False").lower() == "true",
                    symptoms=structured.get("symptoms", []),
                    treatments=structured.get("treatments", "")
                )
                results.append(disease)
            else:
                # Fallback: use raw data without LLM normalization
                logger.warning(f"Using fallback for: {record.get('Name')}")
                symptoms = [s.strip().lower() for s in record.get("Symptoms", "").split(",") if s.strip()]
                disease = StructuredDisease(
                    name=record.get("Name", "").strip().lower(),
                    disease_code=record.get("Disease_Code", ""),
                    chronic=record.get("Chronic", "False").lower() == "true",
                    contagious=record.get("Contagious", "False").lower() == "true",
                    symptoms=symptoms,
                    treatments=record.get("Treatments", "")
                )
                results.append(disease)
        
        return results
    
    async def run(self, csv_path: str, clear_existing: bool = False) -> None:
        # Execute the full ingestion pipeline
        logger.info("Starting Knowledge Graph ingestion pipeline...")
        
        self.graph.create_constraints()
        
        if clear_existing:
            self.graph.clear_graph()
        
        records = self.load_csv(csv_path)
        
        total_processed = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} records)...")
            
            structured_diseases = await self.process_batch(batch)
            
            for disease in structured_diseases:
                self.graph.insert_disease(disease)
                total_processed += 1
            
            if i + BATCH_SIZE < len(records):
                await asyncio.sleep(REQUEST_DELAY)
        
        stats = self.graph.get_statistics()
        logger.info("=" * 50)
        logger.info("Knowledge Graph Construction Complete!")
        logger.info(f"  Diseases: {stats['diseases']}")
        logger.info(f"  Symptoms: {stats['symptoms']}")
        logger.info(f"  HAS_SYMPTOM rels: {stats['has_symptom_relationships']}")
        logger.info("=" * 50)
    
    def close(self) -> None:
        self.graph.close()


async def main():
    csv_path = Path(__file__).parent.parent / "data" / "disease_data.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    pipeline = KGIngestionPipeline()
    
    try:
        await pipeline.run(csv_path=str(csv_path), clear_existing=True)
    finally:
        pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
