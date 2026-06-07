# Generate the labeled benchmark CSV from the source data CSVs.
import os
import csv
import sys
import json
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
from rapidfuzz import fuzz
import ollama

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
DISEASE_CSV = ROOT / "data" / "disease_data.csv"
MEDICINE_CSV = ROOT / "data" / "medicine_data.csv"
OUT_CSV = Path(__file__).parent / "benchmark.csv"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
SEED = 42

# Slot sizes (~100 total). Medicine semantic = description (indication) + side-effects
# only, since that is all the vector DB indexes; named-medicine field lookups are
# known-item exact lookups, not semantic search.
N_DISEASE_SYMPTOMS = 20    # semantic   -> KG symptom retrieval
N_DISEASE_DIFF = 10        # semantic   -> KG, few/ambiguous symptoms
N_DISEASE_NAME = 10        # known_item -> KG disease lookup by name
N_MED_NAME = 15            # known_item -> vector exact name lookup
N_MED_INDICATION = 15      # semantic   -> vector description search (relevance scored)
N_MED_SIDE_EFFECT = 15     # semantic   -> vector side-effects search (relevance scored)
N_HYBRID = 15              # multi_hop  -> symptoms -> disease -> treatment

# Skip ultra-common salts so the task stays non-trivial; cap the stored accept-set.
SALT_MIN, SALT_MAX = 2, 30
ACCEPT_CAP = 50
LEAK_THRESHOLD = 88

SYSTEM = """You are a medical QA dataset author. You write evaluation questions for a
retrieval system that answers strictly from a fixed medical database. For each item you
are given SOURCE DATA (real values from that database) and an INSTRUCTION describing what
to ask. Produce exactly one natural QUESTION and its correct ANSWER.

QUESTION rules:
- Phrase it the way a real person would actually ask (patient, caregiver, curious user).
- Vary wording naturally; do not reuse one fixed template.
- It must be answerable from the SOURCE DATA alone.
- CRITICAL: if the instruction says the target name is HIDDEN, you must NOT mention that
  name (or any obvious variant of it) anywhere in the question. Describe it only by the
  attributes given. This is the whole point of the question.

ANSWER rules:
- Use ONLY facts present in the SOURCE DATA. No outside knowledge, dosages, or advice.
- Be concise and factual; prefer the exact terms used in the data.
- If the data lists several items, include all of them.

Output strict JSON and nothing else: {"question": "...", "answer": "..."}"""

STYLES = [
    "a natural everyday tone",
    "the tone of a concerned patient",
    "a direct, matter-of-fact tone",
    "a brief, conversational tone",
]


def norm(s):
    """Lowercase and collapse whitespace."""
    return " ".join((s or "").lower().split())


def clean_price(raw):
    """Keep only digits and the decimal point from a price string."""
    if raw is None:
        return ""
    raw = str(raw)
    digits = "".join(c for c in raw if c.isdigit() or c == ".")
    return digits or ""


def as_list(raw):
    """Normalize a list-or-comma-string field into a list of trimmed strings."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [s.strip() for s in str(raw or "").split(",") if s.strip()]


def interactions_list(raw):
    """Parse the drug_interactions JSON into 'drug (effect)' strings."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    drugs = data.get("drug", []) or []
    effects = data.get("effect", []) or []
    out = []
    for i, d in enumerate(drugs):
        d = (d or "").strip()
        if not d:
            continue
        eff = (effects[i] if i < len(effects) else "UNKNOWN") or "UNKNOWN"
        out.append(f"{d} ({eff})")
    return out


def leaks(question, names):
    """True if any gold name appears in the question literally or fuzzily."""
    q = norm(question)
    for name in names:
        n = norm(name)
        if not n:
            continue
        if n in q or fuzz.partial_ratio(n, q) >= LEAK_THRESHOLD:
            return True
    return False


def join_gold(names):
    """Join an accept-set of gold names into a |-separated string."""
    return "|".join(names[:ACCEPT_CAP])


def gen(context, instruction, forbid=None, extra_keys=None):
    """Ask Ollama for one {question, answer, ...extras}, retrying and rejecting name leaks."""
    forbid = forbid or []
    extra_keys = extra_keys or []
    hidden_note = ""
    if forbid:
        hidden_note = ("\nThe following name(s) are HIDDEN and must NOT appear in the "
                       f"question: {', '.join(forbid)}.")
    for _ in range(3):
        style = random.choice(STYLES)
        user = (f"SOURCE DATA:\n{context}\n\nINSTRUCTION:\n{instruction}{hidden_note}\n"
                f"Write the question in {style}.\n\nReturn ONLY the JSON object.")
        try:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                format="json",
                options={"temperature": 0.5},
            )
            obj = json.loads(resp["message"]["content"])
            q, a = (obj.get("question") or "").strip(), (obj.get("answer") or "").strip()
        except Exception as e:
            print(f"   gen error: {e}")
            continue
        if not q or not a:
            continue
        if forbid and leaks(q, forbid):
            print(f"   leak rejected: {q[:70]}")
            continue
        out = {"question": q, "answer": a}
        for k in extra_keys:
            out[k] = (obj.get(k) or "").strip()
        return out
    return None


def row(category, subtype, difficulty, retrieval_type, store, retriever,
        question, answer, gold_names, evidence, expected_tools, source,
        score_mode="rank", relevance_field="", relevance_terms=None):
    """Assemble one labeled benchmark row."""
    return dict(category=category, subtype=subtype, difficulty=difficulty,
                retrieval_type=retrieval_type, store=store, retriever=retriever,
                question=question, ground_truth=answer, gold=join_gold(gold_names),
                evidence=evidence, expected_tools=expected_tools, source=source,
                score_mode=score_mode, relevance_field=relevance_field,
                relevance_terms="|".join(relevance_terms or []))


def load_disease_rows():
    """Load disease rows that have both a name and symptoms."""
    rows = []
    with open(DISEASE_CSV, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if (r.get("Name") or "").strip() and (r.get("Symptoms") or "").strip():
                rows.append(r)
    return rows


def disease_symptom_q(drow, difficulty, n_symptoms, hybrid=False):
    """Build a symptom->disease question (or symptom->disease->treatment if hybrid)."""
    name = drow["Name"].strip()
    symptoms = as_list(drow.get("Symptoms", ""))
    if len(symptoms) < 2:
        return None
    shown = symptoms[:n_symptoms]
    treatments = (drow.get("Treatments") or "").strip()
    ctx = (f"Symptoms a patient is experiencing: {', '.join(shown)}\n"
           f"(The condition these point to is: {name} -- HIDDEN, do not name it.)"
           + (f"\nTreatments on record: {treatments}" if hybrid else ""))
    if hybrid:
        instr = ("Write ONE question where a patient describes the symptoms above and asks "
                 "BOTH what the condition might be AND how it is treated. Do not name the condition.")
        cat, sub = "hybrid", "symptom_to_disease_to_treatment"
        tools = "find_diseases_by_symptoms + get_disease_details"
        rtype = "multi_hop"
        evidence = f"disease: {name}; symptoms: {', '.join(shown)}; treatments: {treatments}"
    else:
        instr = ("Write ONE question where a patient describes the symptoms above and asks what "
                 "condition it could be. Do not name the condition.")
        cat = "disease_kg"
        sub = "disease_differential" if difficulty == "hard" else "disease_by_symptoms"
        tools = "find_diseases_by_symptoms"
        rtype = "semantic"
        evidence = f"disease: {name}; symptoms: {', '.join(shown)}"
    qa = gen(ctx, instr, forbid=[name])
    if not qa:
        return None
    return row(cat, sub, difficulty, rtype, "kg", "symptoms",
               qa["question"], qa["answer"], [name], evidence, tools,
               drow.get("Disease_Code", name))


def disease_by_name_q(drow):
    """Build a known-item disease question that names the disease."""
    name = drow["Name"].strip()
    contagious = drow.get("Contagious", "False")
    chronic = drow.get("Chronic", "False")
    symptoms = ", ".join(as_list(drow.get("Symptoms", "")))
    treatments = (drow.get("Treatments") or "").strip()
    sub = random.choice(["contagious", "chronic", "treatment_of_disease", "symptoms_of_disease"])
    ctx = (f"Disease: {name}\nContagious: {contagious}\nChronic: {chronic}\n"
           f"Symptoms: {symptoms or 'n/a'}\nTreatments: {treatments or 'n/a'}")
    instr = {
        "contagious": f"Ask whether {name} is contagious. Answer yes/no from the Contagious field.",
        "chronic": f"Ask whether {name} is a chronic (long-term) condition. Answer from the Chronic field.",
        "treatment_of_disease": f"Ask how {name} is treated. Answer with the treatments.",
        "symptoms_of_disease": f"Ask what the symptoms of {name} are. Answer by listing them.",
    }[sub]
    evidence = {
        "contagious": f"contagious: {contagious}",
        "chronic": f"chronic: {chronic}",
        "treatment_of_disease": f"treatments: {treatments}",
        "symptoms_of_disease": f"symptoms: {symptoms}",
    }[sub]
    qa = gen(ctx, instr)
    if not qa:
        return None
    return row("disease_kg", sub, "easy", "known_item", "kg", "disease_name",
               qa["question"], qa["answer"], [name], evidence, "get_disease_details",
               drow.get("Disease_Code", name))


def build_medicine_index():
    """Stream the medicine CSV into salt->names (accept-sets) and salt->representative row."""
    salt_to_names, salt_to_rep = {}, {}
    with open(MEDICINE_CSV, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            name = (r.get("product_name") or "").strip()
            salt = norm(r.get("salt_composition"))
            if not name or not salt:
                continue
            salt_to_names.setdefault(salt, []).append(name)
            if salt not in salt_to_rep and (r.get("medicine_desc") or "").strip():
                salt_to_rep[salt] = {
                    "product_name": name,
                    "salt_composition": (r.get("salt_composition") or "").strip(),
                    "medicine_desc": (r.get("medicine_desc") or "").strip(),
                    "side_effects": r.get("side_effects") or "",
                    "drug_interactions": r.get("drug_interactions") or "",
                    "manufacturer": (r.get("product_manufactured") or "").strip(),
                    "price": r.get("product_price") or "",
                }
    return salt_to_names, salt_to_rep


def medicine_by_name_q(rec):
    """Build a known-item medicine question (exact name lookup via get_medicine_details)."""
    name = rec["product_name"].strip()
    side = as_list(rec.get("side_effects"))
    inter = interactions_list(rec.get("drug_interactions"))
    salt = (rec.get("salt_composition") or "").strip()
    mfr = (rec.get("manufacturer") or "").strip()
    price = clean_price(rec.get("price"))
    opts = ["composition", "manufacturer_price"]
    if side:
        opts.append("side_effects")
    if inter:
        opts.append("drug_interactions")
    sub = random.choice(opts)
    ctx = (f"Medicine: {name}\nComposition: {salt}\n"
           f"Side effects: {', '.join(side) or 'n/a'}\n"
           f"Interacts with: {', '.join(inter) or 'n/a'}\n"
           f"Manufacturer: {mfr}\nPrice: {price}")
    instr = {
        "composition": f"Ask what the active ingredient/composition of {name} is.",
        "manufacturer_price": f"Ask who manufactures {name} and its price.",
        "side_effects": f"Ask about the side effects of {name}. Answer by listing them.",
        "drug_interactions": f"Ask which drugs {name} interacts with. Answer by listing them.",
    }[sub]
    evidence = {
        "composition": f"salt_composition: {salt}",
        "manufacturer_price": f"manufacturer: {mfr}; price: {price}",
        "side_effects": f"side_effects: {', '.join(side)}",
        "drug_interactions": f"drug_interactions: {', '.join(inter)}",
    }[sub]
    qa = gen(ctx, instr)
    if not qa:
        return None
    return row("medicine_vector", sub, "easy", "known_item", "vector", "medicine_name",
               qa["question"], qa["answer"], [name], evidence, "get_medicine_details", name)


def medicine_by_indication_q(rep, names):
    """Build a semantic description question, scored by whether results cover the condition."""
    name = rep["product_name"].strip()
    salt = (rep.get("salt_composition") or "").strip()
    desc = (rep.get("medicine_desc") or "").strip()[:400]
    if len(desc) < 40:
        return None
    ctx = (f"What the medicine is used for: {desc}\n"
           f"(Brand/product = {name}, active ingredient = {salt}: both HIDDEN.)")
    instr = ("A patient describes what they need a medicine for, based on the use above. "
             "Write ONE question asking what medicine they could take for this purpose. "
             "Do NOT name any brand/product or the active ingredient. "
             "Also output 'condition': the medical condition or use it treats, in 1-4 lowercase "
             "words taken from the text. Answer with a suitable product name.")
    qa = gen(ctx, instr, forbid=names + ([salt] if salt else []), extra_keys=["condition"])
    if not qa or not qa.get("condition"):
        return None
    evidence = f"indication: {desc[:200]}; condition: {qa['condition']}"
    return row("medicine_vector", "by_indication", "hard", "semantic", "vector", "medicine_search",
               qa["question"], qa["answer"], [name], evidence, "search_medicines", name,
               score_mode="relevance", relevance_field="description",
               relevance_terms=[qa["condition"]])


def medicine_by_side_effect_q(rep):
    """Build a semantic side-effect question, scored by whether results list the effect."""
    name = rep["product_name"].strip()
    side = as_list(rep.get("side_effects"))
    if not side:
        return None
    effect = random.choice(side)
    ctx = (f"A medicine is known to cause this side effect: {effect}\n"
           f"(Brand/product = {name}: HIDDEN.)")
    instr = (f"Write ONE question asking which medicine can cause the side effect '{effect}'. "
             "Do NOT name any brand/product. Answer with a suitable product name.")
    qa = gen(ctx, instr, forbid=[name])
    if not qa:
        return None
    evidence = f"side_effect: {effect}"
    return row("medicine_vector", "by_side_effect", "medium", "semantic", "vector", "medicine_search",
               qa["question"], qa["answer"], [name], evidence, "search_medicines", name,
               score_mode="relevance", relevance_field="side_effects",
               relevance_terms=[effect])


def fill(target, makers):
    """Pull from a maker generator until `target` non-None rows are collected."""
    out = []
    for item in makers:
        if item:
            out.append(item)
            print(f"   [{len(out)}/{target}] {item['subtype']:28s} {item['question'][:60]}")
        if len(out) >= target:
            break
    return out


def main():
    """Generate every slot, label the rows, and write the benchmark CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0, help="generate only N of each slot for testing")
    args = ap.parse_args()
    random.seed(SEED)

    def slot(n):
        return args.quick if args.quick else n

    print(f"Model: {OLLAMA_MODEL}")
    print("Loading disease rows...")
    disease_rows = load_disease_rows()
    random.shuffle(disease_rows)

    print("Indexing medicines from medicine_data.csv (streaming the source CSV)...")
    salt_to_names, salt_to_rep = build_medicine_index()
    print(f"  {sum(len(v) for v in salt_to_names.values())} medicines, {len(salt_to_names)} distinct salts")

    # Moderate-size salt groups give meaningful accept-sets: (representative row, names).
    salt_groups = [(salt_to_rep[s], salt_to_names[s]) for s in salt_to_rep
                   if SALT_MIN <= len(salt_to_names[s]) <= SALT_MAX]
    random.shuffle(salt_groups)

    rows = []

    print("\n== Disease by symptoms (semantic, medium) ==")
    di = iter(disease_rows)
    rows += fill(slot(N_DISEASE_SYMPTOMS),
                 (disease_symptom_q(d, "medium", 5) for d in di))

    print("\n== Disease differential (semantic, hard) ==")
    rows += fill(slot(N_DISEASE_DIFF),
                 (disease_symptom_q(d, "hard", 3) for d in di))

    print("\n== Disease by name (known-item, easy) ==")
    rows += fill(slot(N_DISEASE_NAME), (disease_by_name_q(d) for d in di))

    # Shared iterator so name and indication draw distinct salt groups.
    mg = iter(salt_groups)

    print("\n== Medicine by name (known-item, exact lookup) ==")
    rows += fill(slot(N_MED_NAME), (medicine_by_name_q(rep) for rep, _ in mg))

    print("\n== Medicine by indication (semantic description, relevance scored) ==")
    rows += fill(slot(N_MED_INDICATION), (medicine_by_indication_q(rep, names) for rep, names in mg))

    print("\n== Medicine by side effect (semantic, relevance scored) ==")
    se_reps = (rep for rep, _ in salt_groups if as_list(rep.get("side_effects")))
    rows += fill(slot(N_MED_SIDE_EFFECT), (medicine_by_side_effect_q(rep) for rep in se_reps))

    print("\n== Hybrid: symptoms -> disease -> treatment (multi-hop, hard) ==")
    rows += fill(slot(N_HYBRID),
                 (disease_symptom_q(d, "hard", 4, hybrid=True) for d in di))

    for i, r in enumerate(rows, 1):
        r["id"] = i

    cols = ["id", "category", "subtype", "difficulty", "retrieval_type", "store",
            "retriever", "question", "ground_truth", "gold", "evidence",
            "expected_tools", "source", "score_mode", "relevance_field", "relevance_terms"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    by_cat, by_rtype = {}, {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        by_rtype[r["retrieval_type"]] = by_rtype.get(r["retrieval_type"], 0) + 1
    print(f"\nWrote {len(rows)} questions to {OUT_CSV}")
    print(f"  by category       : {by_cat}")
    print(f"  by retrieval_type : {by_rtype}")


if __name__ == "__main__":
    main()
