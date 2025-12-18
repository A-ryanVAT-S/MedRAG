# Kg-Retrieval-Agent for Disease and Symptom Mapping
import os
import json
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jGraphStore

load_dotenv(Path(__file__).parent.parent / ".env")


class KnowledgeGraphRetrievalAgent:
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        try:
            self.graph_store = Neo4jGraphStore(
                username=neo4j_user,
                password=neo4j_password,
                url=neo4j_uri,
                database="neo4j"
            )

            self.graph_store.query("RETURN 1")

            self.connected = True
            print(" [Agent 2] Connected to Neo4j Knowledge Graph")

        except Exception as e:
            print(f" [Agent 2] Failed to connect to Neo4j: {e}")
            self.connected = False
            self.graph_store = None

    def retrieve_by_symptoms(self, symptoms: List[str]) -> Dict:
        if not self.connected:
            return self._get_fallback_response("KG not connected")

        print(f"\n [Agent 2] Querying KG for symptoms: {symptoms}")

        try:
            cypher_query = """
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.name IN $symptoms
            WITH d, COUNT(DISTINCT s) as symptom_count
            ORDER BY symptom_count DESC
            LIMIT 5
            RETURN d.name as disease, 
                   d.treatments as treatments,
                   symptom_count
            """

            results = self.graph_store.query(
                cypher_query,
                param_map={"symptoms": symptoms}
            )

            diseases = []
            for record in results:
                diseases.append({
                    "name": record.get("disease", "Unknown"),
                    "treatments": record.get("treatments", "Not available"),
                    "matched_symptoms": record.get("symptom_count", 0)
                })

            print(f"   Found {len(diseases)} diseases")

            return {
                "diseases": diseases,
                "source": "knowledge_graph",
                "query_type": "symptom_mapping"
            }

        except Exception as e:
            print(f" Error querying KG: {e}")
            return self._get_fallback_response(str(e))

    def retrieve_disease_info(self, disease_name: str) -> Dict:
        if not self.connected:
            return self._get_fallback_response("KG not connected")

        print(f"\n [Agent 2] Querying KG for disease: {disease_name}")

        try:
            cypher_query = """
            MATCH (d:Disease)
            WHERE toLower(d.name) CONTAINS toLower($disease_name)
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
            WITH d, COLLECT(s.name) as symptoms
            RETURN d.name as disease,
                   d.treatments as treatments,
                   symptoms
            LIMIT 1
            """

            results = self.graph_store.query(
                cypher_query,
                param_map={"disease_name": disease_name}
            )

            diseases = []
            for record in results:
                diseases.append({
                    "name": record.get("disease", disease_name),
                    "treatments": record.get("treatments", "Not available"),
                    "symptoms": record.get("symptoms", [])
                })

            print(f"   Retrieved info for: {diseases[0]['name'] if diseases else 'Not found'}")

            return {
                "diseases": diseases,
                "source": "knowledge_graph",
                "query_type": "disease_info"
            }

        except Exception as e:
            print(f"   Error querying KG: {e}")
            return self._get_fallback_response(str(e))

    def _get_fallback_response(self, error_msg: str) -> Dict:
        return {
            "diseases": [],
            "source": "knowledge_graph",
            "error": error_msg,
            "query_type": "fallback"
        }


if __name__ == "__main__":
    agent = KnowledgeGraphRetrievalAgent()

    if agent.connected:
        print("\n" + "=" * 60)
        result = agent.retrieve_by_symptoms(["fever", "cough"])
        print(json.dumps(result, indent=2))

        print("\n" + "=" * 60)
        result = agent.retrieve_disease_info("influenza")
        print(json.dumps(result, indent=2))
