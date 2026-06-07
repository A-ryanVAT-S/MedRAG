# KG retrieval: exact/alias/fuzzy/BM25 candidates -> rerank to seeds -> graph expansion.
import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi

try:
    from .reranker import get_reranker, SHARED_RERANKER_MODEL
except ImportError:
    from reranker import get_reranker, SHARED_RERANKER_MODEL

load_dotenv(Path(__file__).parent.parent / ".env")

FUZZY_CUTOFF = 80          # rapidfuzz score (0-100)
CANDIDATE_POOL = 25        # candidates fed to the reranker
DEFAULT_SEEDS = 5          # seed nodes after reranking


class KnowledgeGraphRetrievalAgent:
    """Retrieves diseases from the Neo4j knowledge graph by symptoms or by name."""

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
        """Connect to Neo4j and build the in-memory symptom/disease indexes."""
        uri = neo4j_uri or os.getenv("NEO4J_URI")
        user = neo4j_user or os.getenv("NEO4J_USERNAME")
        password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        self._reranker = None  # lazy

        # In-memory symptom index (built from graph)
        self.symptom_canon: List[str] = []          # canonical names
        self.alias_to_canon: Dict[str, str] = {}    # alias/name -> canonical
        self.surface_strings: List[str] = []        # names + aliases (for fuzzy/bm25)
        self.surface_to_canon: List[str] = []       # parallel: surface -> canonical
        self.disease_names: List[str] = []
        self._bm25 = None

        self._uri = uri
        self._auth = (user, password)

        try:
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password),
                keep_alive=True,
                max_connection_lifetime=300,
            )
            with self.driver.session(database=self.database) as s:
                s.run("RETURN 1")
            self.connected = True
            print(" [Agent 2] Connected to Neo4j Knowledge Graph")
            self._build_indexes()
        except Exception as e:
            print(f" [Agent 2] Failed to connect to Neo4j: {e}")
            self.connected = False
            self.driver = None

    def _reconnect(self):
        """Re-establish the Neo4j driver, retrying a few times."""
        import time
        for attempt in range(3):
            try:
                if self.driver:
                    try: self.driver.close()
                    except Exception: pass
                self.driver = GraphDatabase.driver(
                    self._uri, auth=self._auth,
                    keep_alive=True,
                    max_connection_lifetime=300,
                )
                with self.driver.session(database=self.database) as s:
                    s.run("RETURN 1")
                print(" [Agent 2] Reconnected to Neo4j")
                self.connected = True
                return
            except Exception as e:
                print(f" [Agent 2] Reconnect attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(3)
        self.connected = False

    def _query(self, cypher: str, **params) -> list:
        """Run a Cypher query, reconnecting once on failure."""
        for attempt in range(2):
            try:
                with self.driver.session(database=self.database) as s:
                    return [r.data() for r in s.run(cypher, **params)]
            except Exception as e:
                if attempt == 0:
                    print(f" [Agent 2] Query error, reconnecting... ({e})")
                    self._reconnect()
                    if not self.connected:
                        return []
                else:
                    print(f" [Agent 2] Query failed after reconnect: {e}")
                    return []
        return []

    def _build_indexes(self):
        """Load symptoms (with aliases) and disease names into in-memory indexes."""
        rows = self._query("MATCH (s:Symptom) RETURN s.name AS name, s.aliases AS aliases")
        for r in rows:
            canon = r["name"]
            self.symptom_canon.append(canon)
            self.alias_to_canon[canon] = canon
            self.surface_strings.append(canon)
            self.surface_to_canon.append(canon)
            for a in (r.get("aliases") or []):
                self.alias_to_canon[a] = canon
                self.surface_strings.append(a)
                self.surface_to_canon.append(canon)

        if self.surface_strings:
            self._bm25 = BM25Okapi([s.split() for s in self.surface_strings])

        self.disease_names = [r["name"] for r in
                              self._query("MATCH (d:Disease) RETURN d.name AS name")]
        print(f"   Indexed {len(self.symptom_canon)} symptoms, {len(self.disease_names)} diseases")

    def _get_reranker(self):
        """Return the shared cross-encoder reranker singleton."""
        return get_reranker(SHARED_RERANKER_MODEL)

    def _candidates_for_term(self, term: str) -> Dict[str, str]:
        """Find candidate symptoms for a term (exact -> alias -> fuzzy -> bm25)."""
        term = term.strip().lower()
        found: Dict[str, str] = {}
        if not term:
            return found

        # exact (canonical) / alias
        if term in self.alias_to_canon:
            method = "exact" if term in self.symptom_canon else "alias"
            found[self.alias_to_canon[term]] = method

        # fuzzy over all surface forms
        for surface, score, idx in process.extract(
            term, self.surface_strings, scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_CUTOFF, limit=10
        ):
            canon = self.surface_to_canon[idx]
            found.setdefault(canon, "fuzzy")

        # bm25 keyword fallback
        if self._bm25 is not None:
            scores = self._bm25.get_scores(term.split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in ranked[:5]:
                if scores[i] <= 0:
                    break
                found.setdefault(self.surface_to_canon[i], "bm25")

        return found

    def _rerank_seeds(self, query: str, candidates: List[str], top_n: int) -> List[tuple]:
        """Rerank candidate symptoms against the query, returning top-N (name, score)."""
        if not candidates:
            return []
        pool = candidates[:CANDIDATE_POOL]
        reranker = self._get_reranker()
        scores = reranker.predict([(query, c) for c in pool])
        ranked = sorted(zip(pool, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def retrieve_by_symptoms(self, symptoms: List[str], two_hop: bool = False) -> Dict:
        """Map symptoms to candidate diseases via seed reranking and graph expansion."""
        if not self.connected:
            return self._fallback("KG not connected")
        if not self.symptom_canon:
            return self._fallback("KG has no symptom nodes (run ingestion_kg.py)")

        print(f"\n [Agent 2] Symptom retrieval: {symptoms}")
        query = ", ".join(symptoms)

        candidates: Dict[str, str] = {}
        for term in symptoms:
            for canon, method in self._candidates_for_term(term).items():
                candidates.setdefault(canon, method)

        if not candidates:
            return self._fallback("No matching symptoms in KG")

        seeds = self._rerank_seeds(query, list(candidates), DEFAULT_SEEDS)
        seed_names = [name for name, _ in seeds]
        print(f"   Seeds: {seed_names}")

        diseases = self._expand_from_symptoms(seed_names, two_hop)
        return {
            "diseases": diseases,
            "source": "knowledge_graph",
            "query_type": "symptom_mapping",
            "seeds": [{"symptom": n, "score": round(s, 3)} for n, s in seeds],
        }

    def retrieve_disease_info(self, disease_name: str, two_hop: bool = False) -> Dict:
        """Return the full record for a single named disease."""
        if not self.connected:
            return self._fallback("KG not connected")

        print(f"\n [Agent 2] Disease retrieval: {disease_name}")
        target = self._match_disease_name(disease_name)
        if not target:
            return self._fallback(f"Disease '{disease_name}' not found in KG")

        diseases = self._expand_disease(target, two_hop)
        return {
            "diseases": diseases,
            "source": "knowledge_graph",
            "query_type": "disease_info",
        }

    def _expand_from_symptoms(self, seed_names: List[str], two_hop: bool) -> List[Dict]:
        """Find diseases sharing the seed symptoms, ranked by match count."""
        rows = self._query("""
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.name IN $seeds
            WITH d, count(DISTINCT s) AS matched
            ORDER BY matched DESC
            LIMIT 5
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(alls:Symptom)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
            RETURN d.name AS name, d.disease_code AS code,
                   d.contagious AS contagious, d.chronic AS chronic,
                   d.raw_treatments AS raw_treatments,
                   collect(DISTINCT alls.name) AS symptoms,
                   collect(DISTINCT t.name) AS treatments,
                   matched
        """, seeds=seed_names)
        diseases = [self._format_disease(r, matched=r.get("matched")) for r in rows]

        if two_hop and diseases:
            diseases = self._attach_related(diseases)
        print(f"   Expanded to {len(diseases)} diseases")
        return diseases

    def _expand_disease(self, name: str, two_hop: bool) -> List[Dict]:
        """Fetch one disease node with its symptoms and treatments."""
        rows = self._query("""
            MATCH (d:Disease {name: $name})
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
            RETURN d.name AS name, d.disease_code AS code,
                   d.contagious AS contagious, d.chronic AS chronic,
                   d.raw_treatments AS raw_treatments,
                   collect(DISTINCT s.name) AS symptoms,
                   collect(DISTINCT t.name) AS treatments
        """, name=name)
        diseases = [self._format_disease(r) for r in rows]
        if two_hop and diseases:
            diseases = self._attach_related(diseases)
        return diseases

    def _attach_related(self, diseases: List[Dict]) -> List[Dict]:
        """Attach 2-hop related diseases that share a treatment."""
        for d in diseases:
            if not d.get("treatments"):
                continue
            related = self._query("""
                MATCH (d:Disease {name: $name})-[:TREATED_BY]->(t:Treatment)<-[:TREATED_BY]-(other:Disease)
                WHERE other.name <> $name
                RETURN DISTINCT other.name AS name, collect(DISTINCT t.name) AS shared_treatments
                LIMIT 5
            """, name=d["name"])
            d["related_diseases"] = related
        return diseases

    def _match_disease_name(self, query: str) -> Optional[str]:
        """Resolve a query to a canonical disease name (exact -> substring -> fuzzy)."""
        q = query.strip().lower()
        if q in self.disease_names:
            return q
        for n in self.disease_names:
            if q in n or n in q:
                return n
        match = process.extractOne(q, self.disease_names, scorer=fuzz.token_sort_ratio,
                                   score_cutoff=FUZZY_CUTOFF)
        return match[0] if match else None

    @staticmethod
    def _format_disease(r: Dict, matched: Optional[int] = None) -> Dict:
        """Shape a raw query row into a clean disease dict."""
        out = {
            "name": r.get("name", "Unknown"),
            "symptoms": [s for s in (r.get("symptoms") or []) if s],
            "treatments": [t for t in (r.get("treatments") or []) if t],
            "raw_treatments": r.get("raw_treatments") or "",
            "contagious": r.get("contagious"),
            "chronic": r.get("chronic"),
        }
        if matched is not None:
            out["matched_symptoms"] = matched
        return out

    def _fallback(self, error_msg: str) -> Dict:
        """Return an empty result carrying an error message."""
        return {"diseases": [], "source": "knowledge_graph", "error": error_msg, "query_type": "fallback"}

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    agent = KnowledgeGraphRetrievalAgent()
    if agent.connected:
        print(json.dumps(agent.retrieve_by_symptoms(["fever", "cough"]), indent=2, ensure_ascii=False)[:1500])
        print(json.dumps(agent.retrieve_disease_info("influenza"), indent=2, ensure_ascii=False)[:1500])
        agent.close()
