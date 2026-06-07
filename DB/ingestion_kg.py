# KG ingestion: normalize diseases with a local LLM, resolve synonymous symptoms
# and treatments into canonical nodes, then write Disease/Symptom/Treatment to Neo4j.
import os
import csv
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rapidfuzz import fuzz

# Logging: console (INFO) + timestamped file (DEBUG, full detail of what got ingested)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"kg_ingestion_{_run_stamp}.log"
REPORT_FILE = LOG_DIR / f"kg_ingestion_report_{_run_stamp}.txt"

logger = logging.getLogger("kg_ingestion")
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _console = logging.StreamHandler()
    _console.setLevel(logging.INFO)
    _console.setFormatter(_fmt)
    _file = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _file.setLevel(logging.DEBUG)
    _file.setFormatter(_fmt)
    logger.addHandler(_console)
    logger.addHandler(_file)
    logger.info(f"Logging to {LOG_FILE}")

load_dotenv(Path(__file__).parent.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")  # Aura default db is "neo4j"

# Local model for normalization; override with the OLLAMA_MODEL env var.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Fuzzy thresholds (0-100) for merging synonymous entities.
SYMPTOM_MERGE_THRESHOLD = 90
TREATMENT_MERGE_THRESHOLD = 90

CACHE_FILE = Path(__file__).parent / ".llm_cache.json"


@dataclass
class DiseaseRecord:
    disease_code: str
    name: str
    contagious: bool
    chronic: bool
    raw_treatments: str
    # entity = {"name": str, "aliases": [str]}
    symptoms: list[dict] = field(default_factory=list)
    treatments: list[dict] = field(default_factory=list)


class OllamaNormalizer:
    """Atomize and normalize a disease's free-text symptoms/treatments with a local LLM."""

    SYSTEM_PROMPT = """You are a medical knowledge-graph normalization assistant.
You convert ONE disease's free-text symptoms and treatments into clean structured JSON
that will be inserted into a graph database.

THE GRAPH IS MANY-TO-MANY:
- One free-text field becomes MANY atomic nodes (split it apart).
- The SAME symptom or treatment can belong to MANY diseases. You MUST give a shared
  concept the SAME canonical "name" every time so it collapses into ONE shared node.
  "fever" must be named "fever" for every disease - never "high fever" for one and
  "mild fever" for another.

RULES:

1. ATOMIZE - split every compound field into individual items.
   "nausea, vomiting, abdominal pain" -> three symptom nodes.
   "rest, fluids, antiviral medication" -> three treatment nodes.

2. SPLIT "X or Y" AND "X and Y" COMPOUNDS into two separate nodes each.
   "nausea or vomiting" -> two nodes: "nausea" and "vomiting".
   "weakness or numbness" -> two nodes: "weakness" and "numbness".
   "shooting or burning pain" -> two nodes: "shooting pain" and "burning pain".

3. EXPAND "X (such as A, B, C)" or "X (e.g. A, B)" by creating A, B, C as
   INDIVIDUAL nodes. Drop the generic parent X if A/B/C are specific enough.
   "medications (such as antidepressants, anticonvulsants)" ->
     three nodes: "antidepressants", "anticonvulsants". Drop "medications".
   "surgery (such as craniotomy, burr hole evacuation)" ->
     one node: "surgery" with aliases ["craniotomy", "burr hole evacuation"].
   Rule of thumb: if the children are specific named therapies/drugs, use them.
   If they are vague examples, keep the parent and list children as aliases.

4. CANONICALIZE the name to the shortest standard medical term (MAX 4 WORDS).
   - Strip severity/frequency/location qualifiers from the NAME; put them in aliases.
     "High fever" -> name "fever", alias "high fever".
     "intense itching at night" -> name "itching", alias "intense itching".
   - DO NOT over-generalize: keep the primary distinguishing quality in the name.
     "burning pain" stays "burning pain" - NOT just "pain".
     "abdominal pain" stays "abdominal pain" - NOT just "pain".
     "dry cough" stays "cough" with alias "dry cough" - core word is enough.
   - Surgery terminology: ALWAYS use "surgery" as canonical.
     "surgical intervention", "surgical repair", "surgical procedure" -> name "surgery".

5. ALIASES = well-known medical synonyms + original surface wordings you stripped.
   Lowercase. 2-4 aliases max. Empty list [] if none.
   Do NOT invent obscure or uncertain terms.

6. DISTINGUISH "X and Y" carefully.
   Split ONLY when X and Y are each complete independent medical concepts.
   "nausea and vomiting" -> two nodes (both are independent symptoms).
   "close monitoring of vital signs and neurological status" -> ONE node "monitoring"
     (vital signs / neurological status are what is monitored, not separate treatments).
   "counseling and support" -> ONE node "counseling and support" (single service).

7. STRICT - no hallucination. Output ONLY symptoms and treatments explicitly stated
   in the input text. Never add, infer, or assume anything not present.

8. Aliases must be DIFFERENT from the name. Never repeat the name as an alias.
   Bad: name "headache", aliases ["headache"]. Good: aliases ["cephalgia"].

9. Lowercase all names and aliases. Trim whitespace. Remove duplicates and empties.

10. Empty symptoms/treatments field -> return [].

11. Output ONLY the JSON object. No commentary, no markdown, no code fences.

JSON schema (follow exactly):
{
  "disease_name": "normalized disease name, lowercase",
  "symptoms":   [{"name": "canonical symptom", "aliases": ["synonym1", "synonym2"]}],
  "treatments": [{"name": "canonical treatment", "aliases": ["synonym1"]}]
}"""

    # Few-shot turns teaching atomization, canonical naming, and shared nodes.
    FEW_SHOT = [
        # Shot 1: qualifier stripping (high fever -> fever), shared-node concept
        {"role": "user", "content": (
            "Disease Name: Influenza\n"
            "Symptoms: High fever, dry cough, body aches and severe fatigue\n"
            "Treatments: Rest, plenty of fluids, antiviral medication (in severe cases)\n\n"
            "Return ONLY the JSON object."
        )},
        {"role": "assistant", "content": json.dumps({
            "disease_name": "influenza",
            "symptoms": [
                {"name": "fever",     "aliases": ["high fever", "pyrexia"]},
                {"name": "cough",     "aliases": ["dry cough"]},
                {"name": "body ache", "aliases": ["myalgia", "body aches"]},
                {"name": "fatigue",   "aliases": ["tiredness", "severe fatigue"]},
            ],
            "treatments": [
                {"name": "rest",                 "aliases": []},
                {"name": "fluids",               "aliases": ["hydration", "oral rehydration"]},
                {"name": "antiviral medication", "aliases": ["antivirals", "oseltamivir"]},
            ],
        }, ensure_ascii=False)},

        # Shot 2: fever/fatigue reuse the same canonical names as shot 1 (shared nodes)
        {"role": "user", "content": (
            "Disease Name: Malaria\n"
            "Symptoms: Recurrent fever with chills, sweating, headache and fatigue\n"
            "Treatments: Antimalarial drugs such as chloroquine or artemisinin, supportive care\n\n"
            "Return ONLY the JSON object."
        )},
        {"role": "assistant", "content": json.dumps({
            "disease_name": "malaria",
            "symptoms": [
                {"name": "fever",    "aliases": ["pyrexia", "recurrent fever"]},
                {"name": "chills",   "aliases": ["shivering", "rigors"]},
                {"name": "sweating", "aliases": ["diaphoresis"]},
                {"name": "headache", "aliases": ["cephalgia"]},
                {"name": "fatigue",  "aliases": ["tiredness"]},
            ],
            "treatments": [
                {"name": "chloroquine",    "aliases": ["chloroquine phosphate"]},
                {"name": "artemisinin",    "aliases": ["artemisinin-based therapy"]},
                {"name": "supportive care","aliases": []},
            ],
        }, ensure_ascii=False)},

        # Shot 3: "X or Y" splits, "and" within one concept stays joined, surgery canonical
        {"role": "user", "content": (
            "Disease Name: Subdural Hemorrhage\n"
            "Symptoms: Headache, confusion, dizziness, nausea or vomiting, "
            "seizures, weakness or numbness\n"
            "Treatments: Immediate medical attention, close monitoring of vital signs "
            "and neurological status, diagnostic imaging (such as CT or MRI scan), "
            "potential surgical intervention (such as craniotomy or burr hole evacuation), "
            "medication to manage symptoms\n\n"
            "Return ONLY the JSON object."
        )},
        {"role": "assistant", "content": json.dumps({
            "disease_name": "subdural hemorrhage",
            "symptoms": [
                {"name": "headache",       "aliases": ["cephalgia"]},
                {"name": "confusion",      "aliases": ["disorientation", "altered mental status"]},
                {"name": "dizziness",      "aliases": ["vertigo"]},
                {"name": "nausea",         "aliases": ["emesis"]},
                {"name": "vomiting",       "aliases": []},
                {"name": "seizures",       "aliases": ["convulsions", "epileptic episode"]},
                {"name": "weakness",       "aliases": ["limb weakness", "hemiparesis"]},
                {"name": "numbness",       "aliases": ["hypoesthesia"]},
            ],
            "treatments": [
                {"name": "medical attention",  "aliases": ["emergency care"]},
                {"name": "monitoring",         "aliases": ["vital signs monitoring", "neurological monitoring"]},
                {"name": "diagnostic imaging", "aliases": ["ct scan", "mri scan"]},
                {"name": "surgery",            "aliases": ["craniotomy", "burr hole evacuation"]},
                {"name": "medication",         "aliases": ["symptom management"]},
            ],
        }, ensure_ascii=False)},

        # Shot 4: "such as A, B" expands to individual nodes, generic parent dropped
        {"role": "user", "content": (
            "Disease Name: Neuropathic Pain\n"
            "Symptoms: Shooting or burning pain, tingling or numbness\n"
            "Treatments: Medications (such as antidepressants, anticonvulsants), "
            "nerve blocks, physical therapy\n\n"
            "Return ONLY the JSON object."
        )},
        {"role": "assistant", "content": json.dumps({
            "disease_name": "neuropathic pain",
            "symptoms": [
                {"name": "shooting pain", "aliases": ["lancinating pain"]},
                {"name": "burning pain",  "aliases": ["burning sensation", "causalgia"]},
                {"name": "tingling",      "aliases": ["paresthesia", "pins and needles"]},
                {"name": "numbness",      "aliases": ["hypoesthesia"]},
            ],
            "treatments": [
                {"name": "antidepressants", "aliases": ["tricyclic antidepressants", "snri"]},
                {"name": "anticonvulsants", "aliases": ["gabapentin", "pregabalin"]},
                {"name": "nerve blocks",    "aliases": ["nerve block injection"]},
                {"name": "physical therapy","aliases": ["physiotherapy"]},
            ],
        }, ensure_ascii=False)},
    ]

    def __init__(self, model: str):
        """Set up the Ollama client and load the normalization cache."""
        import ollama
        self.client = ollama
        self.model = model
        self.cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached normalizations from disk if present."""
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                logger.info(f"Loaded {len(self.cache)} cached normalizations")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
                self.cache = {}

    def _save_cache(self):
        """Persist the normalization cache to disk."""
        try:
            CACHE_FILE.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _key(self, name: str, symptoms: str, treatments: str) -> str:
        """Build a cache key from the model and the raw record fields."""
        return hashlib.md5(f"{self.model}|{name}|{symptoms}|{treatments}".encode()).hexdigest()

    def normalize(self, name: str, symptoms_raw: str, treatments_raw: str) -> Optional[dict]:
        """Normalize one disease's symptoms/treatments to structured JSON (cached)."""
        key = self._key(name, symptoms_raw, treatments_raw)
        if key in self.cache:
            return self.cache[key]

        user_prompt = (
            f"Disease Name: {name}\n"
            f"Symptoms: {symptoms_raw}\n"
            f"Treatments: {treatments_raw}\n\n"
            "Return ONLY the JSON object."
        )
        messages = (
            [{"role": "system", "content": self.SYSTEM_PROMPT}]
            + self.FEW_SHOT
            + [{"role": "user", "content": user_prompt}]
        )
        try:
            resp = self.client.chat(
                model=self.model,
                messages=messages,
                format="json",  # Ollama JSON mode
                options={"temperature": 0},
            )
            content = resp["message"]["content"].strip()
            result = json.loads(content)
            self.cache[key] = result
            self._save_cache()
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama error for '{name}': {e}")
            raise  # connection errors should stop the run (handled in pipeline)


def _clean_entity_list(raw: list) -> list[dict]:
    """Coerce LLM output into [{name, aliases[]}], lowercased and deduped."""
    out, seen = [], set()
    for item in raw or []:
        if isinstance(item, str):
            name, aliases = item, []
        elif isinstance(item, dict):
            name, aliases = item.get("name", ""), item.get("aliases", []) or []
        else:
            continue
        name = str(name).strip().lower().replace("_", " ")
        if not name or name in seen:
            continue
        seen.add(name)
        clean_aliases = sorted({str(a).strip().lower().replace("_", " ")
                                for a in aliases if str(a).strip()})
        out.append({"name": name, "aliases": clean_aliases})
    return out


def naive_fallback(name: str, symptoms_raw: str, treatments_raw: str) -> dict:
    """Comma-split fallback used only when a record fails LLM normalization."""
    split = lambda s: [{"name": p.strip().lower(), "aliases": []}
                       for p in s.split(",") if p.strip()]
    return {
        "disease_name": name.strip().lower(),
        "symptoms": split(symptoms_raw),
        "treatments": split(treatments_raw),
    }


class EntityResolver:
    """Greedy fuzzy clustering of names into canonical entities carrying aliases."""

    def __init__(self, threshold: int):
        """Set the merge threshold and init the canonical/lookup maps."""
        self.threshold = threshold
        self.canonical: dict[str, set] = {}   # canonical_name -> set of surface forms
        self.lookup: dict[str, str] = {}      # surface form -> canonical_name

    def resolve(self, name: str, aliases: list[str]) -> str:
        """Register an entity (merging into a fuzzy match) and return its canonical name."""
        name = name.strip().lower().replace("_", " ")
        if not name:
            return ""

        if name in self.lookup:
            canon = self.lookup[name]
        else:
            canon = self._find_fuzzy_match(name)
            if canon is None:
                canon = name
                self.canonical[canon] = set()
            self.canonical[canon].add(name)
            self.lookup[name] = canon

        for a in aliases:
            a = a.strip().lower().replace("_", " ")
            if a and a != canon:
                self.canonical[canon].add(a)
                self.lookup.setdefault(a, canon)
        return canon

    def _find_fuzzy_match(self, name: str) -> Optional[str]:
        """Return the best canonical name above the threshold, or None."""
        best, best_score = None, 0
        for canon in self.canonical:
            score = fuzz.token_sort_ratio(name, canon)
            if score > best_score:
                best, best_score = canon, score
        return best if best_score >= self.threshold else None

    def aliases_for(self, canonical_name: str) -> list[str]:
        """Return all surface forms except the canonical name itself."""
        return sorted(self.canonical.get(canonical_name, set()) - {canonical_name})


class Neo4jKG:
    """Thin Neo4j writer with reconnect/retry for the graph upserts."""

    def __init__(self, uri, user, password, database):
        """Store connection settings; the driver opens lazily in connect()."""
        self.uri = uri
        self.auth = (user, password)
        self.database = database
        self.driver = None

    def connect(self):
        """Open a fresh connection right before writes to avoid idle timeouts."""
        import time
        for attempt in range(3):
            try:
                if self.driver:
                    try:
                        self.driver.close()
                    except Exception:
                        pass
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=self.auth,
                    keep_alive=True,
                    max_connection_lifetime=300,
                    connection_acquisition_timeout=30,
                )
                with self.driver.session(database=self.database) as s:
                    s.run("RETURN 1")
                logger.info("Neo4j connection established")
                return
            except Exception as e:
                logger.warning(f"Neo4j connect attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError("Could not connect to Neo4j after 3 attempts")

    def _run_with_retry(self, cypher: str, max_retries: int = 3, **params):
        """Run a Cypher statement, reconnecting and retrying on stale connections."""
        import time
        for attempt in range(max_retries):
            try:
                with self.driver.session(database=self.database) as s:
                    s.run(cypher, **params)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Neo4j write error (attempt {attempt+1}): {e}, reconnecting...")
                    time.sleep(3)
                    self.connect()
                else:
                    raise

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()

    def setup(self, clear: bool):
        """Optionally clear the graph, then create uniqueness constraints."""
        if clear:
            self._run_with_retry("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared")
        self._run_with_retry("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.disease_code IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE")
        logger.info("Constraints ready")

    def upsert_symptom_nodes(self, symptoms: list[dict]):
        """Upsert Symptom nodes in batches of 500 (Aura Bolt size limit)."""
        for i in range(0, len(symptoms), 500):
            batch = symptoms[i:i + 500]
            self._run_with_retry("""
                UNWIND $rows AS row
                MERGE (n:Symptom {name: row.name})
                SET n.aliases = row.aliases
            """, rows=batch)

    def upsert_treatment_nodes(self, treatments: list[dict]):
        """Upsert Treatment nodes in batches of 500."""
        for i in range(0, len(treatments), 500):
            batch = treatments[i:i + 500]
            self._run_with_retry("""
                UNWIND $rows AS row
                MERGE (n:Treatment {name: row.name})
                SET n.aliases = row.aliases
            """, rows=batch)

    def upsert_disease(self, disease: DiseaseRecord, symptom_names: list[str], treatment_names: list[str]):
        """Upsert one Disease node and its symptom/treatment relationships."""
        self._run_with_retry("""
            MERGE (d:Disease {disease_code: $code})
            SET d.name = $name,
                d.contagious = $contagious,
                d.chronic = $chronic,
                d.raw_treatments = $raw_treatments
            WITH d
            UNWIND $symptoms AS sname
            MATCH (sym:Symptom {name: sname})
            MERGE (d)-[:HAS_SYMPTOM]->(sym)
        """, code=disease.disease_code, name=disease.name,
             contagious=disease.contagious, chronic=disease.chronic,
             raw_treatments=disease.raw_treatments, symptoms=symptom_names)

        if treatment_names:
            self._run_with_retry("""
                MATCH (d:Disease {disease_code: $code})
                UNWIND $treatments AS tname
                MATCH (t:Treatment {name: tname})
                MERGE (d)-[:TREATED_BY]->(t)
            """, code=disease.disease_code, treatments=treatment_names)

    def stats(self) -> dict:
        """Count nodes and relationships currently in the graph."""
        with self.driver.session(database=self.database) as s:
            r = s.run("""
                OPTIONAL MATCH (d:Disease)   WITH count(DISTINCT d) AS diseases
                OPTIONAL MATCH (sy:Symptom)  WITH diseases, count(DISTINCT sy) AS symptoms
                OPTIONAL MATCH (t:Treatment) WITH diseases, symptoms, count(DISTINCT t) AS treatments
                OPTIONAL MATCH ()-[h:HAS_SYMPTOM]->() WITH diseases, symptoms, treatments, count(h) AS hs
                OPTIONAL MATCH ()-[tb:TREATED_BY]->()
                RETURN diseases, symptoms, treatments, hs AS has_symptom, count(tb) AS treated_by
            """).single()
            return dict(r)


class KGIngestionPipeline:
    """Normalize, resolve, and write the disease CSV into the Neo4j knowledge graph."""

    def __init__(self):
        """Build the normalizer, entity resolvers, and (lazy) Neo4j writer."""
        self.normalizer = OllamaNormalizer(OLLAMA_MODEL)
        self.symptom_resolver = EntityResolver(SYMPTOM_MERGE_THRESHOLD)
        self.treatment_resolver = EntityResolver(TREATMENT_MERGE_THRESHOLD)
        self.kg = Neo4jKG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

    def load_csv(self, path: str) -> list[dict]:
        """Read the disease CSV into a list of row dicts."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        logger.info(f"Loaded {len(rows)} disease rows")
        return rows

    def _check_ollama(self):
        """Verify Ollama is reachable and the model is available."""
        try:
            import ollama
            available = [m.get("model", "") for m in ollama.list().get("models", [])]
            if not any(self.normalizer.model.split(":")[0] in a for a in available):
                logger.warning(
                    f"Model '{self.normalizer.model}' not found in Ollama. "
                    f"Run:  ollama pull {self.normalizer.model}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama ({e}). Install from https://ollama.com, then "
                f"`ollama pull {self.normalizer.model}` and ensure `ollama serve` is running."
            )

    def run(self, csv_path: str, clear_existing: bool = True):
        """Run the full pipeline: normalize -> resolve entities -> write the graph."""
        logger.info("Starting KG ingestion...")
        self._check_ollama()
        rows = self.load_csv(csv_path)

        # Phase 1: normalize each disease with the local model.
        diseases: list[DiseaseRecord] = []
        for i, row in enumerate(rows, 1):
            name = row.get("Name", "")
            symptoms_raw = row.get("Symptoms", "")
            treatments_raw = row.get("Treatments", "")

            structured = self.normalizer.normalize(name, symptoms_raw, treatments_raw)
            if not structured:
                logger.warning(f"Fallback normalization for: {name}")
                structured = naive_fallback(name, symptoms_raw, treatments_raw)

            diseases.append(DiseaseRecord(
                disease_code=(row.get("Disease_Code", "") or "").strip().lower(),
                name=(structured.get("disease_name") or name).strip().lower(),
                contagious=str(row.get("Contagious", "False")).strip().lower() == "true",
                chronic=str(row.get("Chronic", "False")).strip().lower() == "true",
                raw_treatments=treatments_raw.strip(),
                symptoms=_clean_entity_list(structured.get("symptoms")),
                treatments=_clean_entity_list(structured.get("treatments")),
            ))
            if i % 25 == 0:
                logger.info(f"Normalized {i}/{len(rows)}")

        # Phase 2: resolve symptoms/treatments into canonical nodes across all diseases.
        logger.info("Resolving entities (global dedup)...")
        disease_symptom_canon: dict[str, list[str]] = {}
        disease_treatment_canon: dict[str, list[str]] = {}
        for d in diseases:
            s_names = []
            for ent in d.symptoms:
                canon = self.symptom_resolver.resolve(ent["name"], ent["aliases"])
                if canon and canon not in s_names:
                    s_names.append(canon)
            disease_symptom_canon[d.disease_code] = s_names

            t_names = []
            for ent in d.treatments:
                canon = self.treatment_resolver.resolve(ent["name"], ent["aliases"])
                if canon and canon not in t_names:
                    t_names.append(canon)
            disease_treatment_canon[d.disease_code] = t_names

            # Per-disease detail goes to the log file only.
            logger.debug(
                f"[{d.disease_code}] {d.name} (contagious={d.contagious}, chronic={d.chronic})\n"
                f"    symptoms  -> {s_names}\n"
                f"    treatments-> {t_names}"
            )

        symptom_nodes = [{"name": n, "aliases": self.symptom_resolver.aliases_for(n)}
                         for n in self.symptom_resolver.canonical]
        treatment_nodes = [{"name": n, "aliases": self.treatment_resolver.aliases_for(n)}
                           for n in self.treatment_resolver.canonical]
        logger.info(f"Canonical: {len(symptom_nodes)} symptoms, {len(treatment_nodes)} treatments")

        # Phase 3: connect (after normalization) and write the graph.
        logger.info("Connecting to Neo4j...")
        self.kg.connect()
        self.kg.setup(clear=clear_existing)
        self.kg.upsert_symptom_nodes(symptom_nodes)
        self.kg.upsert_treatment_nodes(treatment_nodes)
        for d in diseases:
            self.kg.upsert_disease(
                d,
                disease_symptom_canon.get(d.disease_code, []),
                disease_treatment_canon.get(d.disease_code, []),
            )

        stats = self.kg.stats()
        logger.info("Knowledge graph complete")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        self._write_report(diseases, disease_symptom_canon, disease_treatment_canon,
                           symptom_nodes, treatment_nodes, stats)
        logger.info(f"Ingestion report written to {REPORT_FILE}")

    def _write_report(self, diseases, sym_canon, treat_canon, symptom_nodes, treatment_nodes, stats):
        """Write a human-readable record of exactly what landed in the graph."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"MedRAG KG Ingestion Report  ({_run_stamp})")
        lines.append(f"Model: {OLLAMA_MODEL}")
        lines.append("=" * 70)
        lines.append("STATS: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
        lines.append("")

        lines.append("-" * 70)
        lines.append("DISEASES")
        lines.append("-" * 70)
        for d in diseases:
            lines.append(f"[{d.disease_code}] {d.name}  "
                         f"(contagious={d.contagious}, chronic={d.chronic})")
            lines.append(f"    symptoms : {', '.join(sym_canon.get(d.disease_code, [])) or '-'}")
            lines.append(f"    treatments: {', '.join(treat_canon.get(d.disease_code, [])) or '-'}")
            if d.raw_treatments:
                lines.append(f"    raw_treatments: {d.raw_treatments}")
            lines.append("")

        lines.append("-" * 70)
        lines.append(f"CANONICAL SYMPTOM NODES ({len(symptom_nodes)})")
        lines.append("-" * 70)
        for n in sorted(symptom_nodes, key=lambda x: x["name"]):
            alias = f"  <- aliases: {', '.join(n['aliases'])}" if n["aliases"] else ""
            lines.append(f"  {n['name']}{alias}")

        lines.append("")
        lines.append("-" * 70)
        lines.append(f"CANONICAL TREATMENT NODES ({len(treatment_nodes)})")
        lines.append("-" * 70)
        for n in sorted(treatment_nodes, key=lambda x: x["name"]):
            alias = f"  <- aliases: {', '.join(n['aliases'])}" if n["aliases"] else ""
            lines.append(f"  {n['name']}{alias}")

        REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    def close(self):
        """Close the underlying Neo4j connection."""
        self.kg.close()


def main():
    """Entry point: ingest data/disease_data.csv into the knowledge graph."""
    csv_path = Path(__file__).parent.parent / "data" / "disease_data.csv"
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return
    pipeline = KGIngestionPipeline()
    try:
        pipeline.run(str(csv_path), clear_existing=True)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
