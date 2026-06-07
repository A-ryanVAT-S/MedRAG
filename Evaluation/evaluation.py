# Score the agents on the benchmark for retrieval, routing, and answer quality.
import os
import csv
import sys
import json
import time
import math
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rapidfuzz import fuzz
from groq import Groq

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent2_kg_retrieval import KnowledgeGraphRetrievalAgent
from agents.agent3_vector_retrieval import VectorRetrievalAgent
from agents.agent_core import MedicalAgent

BENCHMARK = Path(__file__).parent / "benchmark.csv"
OUT_JSON = Path(__file__).parent / "eval.json"
# Per-query results stream here so a crash mid-run keeps everything scored so far.
PARTIAL_JSONL = Path(__file__).parent / "eval_partial.jsonl"

JUDGE_MODEL = "llama-3.1-8b-instant"
MATCH_THRESHOLD = 85


def norm(s):
    """Lowercase and collapse whitespace."""
    return " ".join((s or "").lower().split())


def same_item(gold, item):
    """True if two names refer to the same item (substring or fuzzy match)."""
    g, it = norm(gold), norm(item)
    if not g or not it:
        return False
    if g == it or g in it or it in g:
        return True
    return fuzz.token_set_ratio(g, it) >= MATCH_THRESHOLD


def rank_of(golds, ranked):
    """1-indexed rank of the first retrieved item matching any gold, else None."""
    if isinstance(golds, str):
        golds = [golds]
    seen, pos = set(), 0
    for item in ranked:
        key = norm(item)
        if key in seen:
            continue
        seen.add(key)
        pos += 1
        if any(same_item(g, item) for g in golds):
            return pos
    return None


def parse_evidence(s):
    """Parse a 'key: val; key: val' evidence string into a dict."""
    out = {}
    for part in (s or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def pct(values, p):
    """Percentile p (0-1) of a list of numbers."""
    if not values:
        return 0.0
    v = sorted(values)
    i = min(len(v) - 1, int(round(p * (len(v) - 1))))
    return round(v[i], 2)


def stats(values):
    """Mean / p50 / p95 / min / max of a list of numbers."""
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    return {"mean": round(sum(values) / len(values), 2), "p50": pct(values, 0.50),
            "p95": pct(values, 0.95), "min": round(min(values), 2), "max": round(max(values), 2)}


def golds_for_row(row):
    """Read the |-separated accept-set of gold names from a benchmark row."""
    return [g.strip() for g in (row.get("gold") or "").split("|") if g.strip()]


def answer_items_for_row(row):
    """Flatten the evidence values into individual items for answer-recall scoring."""
    items = []
    for v in parse_evidence(row.get("evidence", "")).values():
        items += [x.strip() for x in v.split(",") if x.strip()]
    return items


def relevance_terms_for_row(row):
    """Read the |-separated relevance terms (for relevance-scored questions)."""
    return [t.strip() for t in (row.get("relevance_terms") or "").split("|") if t.strip()]


def relevance_rank(items, field, terms):
    """1-indexed rank of the first item whose `field` covers any term, else None.

    A term matches if it is a substring of the field, or if every word in it
    (length >= 4) appears in the field. Used to score semantic medicine questions
    where any medicine that genuinely covers the indication / side effect counts.
    """
    terms = [norm(t) for t in terms if t.strip()]
    if not terms:
        return None
    seen, pos = set(), 0
    for it in items:
        key = norm(it.get("name"))
        if key in seen:
            continue
        seen.add(key)
        pos += 1
        hay = it.get(field)
        if isinstance(hay, list):
            hay = " ".join(str(x) for x in hay)
        hay = norm(hay)
        for t in terms:
            toks = [w for w in t.split() if len(w) >= 4]
            if t in hay or (toks and all(w in hay for w in toks)):
                return pos
    return None


def retrieve_direct(kg, vec, retriever, question, name=""):
    """Run the real retriever (no LLM) and return (item dicts, ms).

    `name` is the exact medicine name used for the known-item lookup path; every
    other path is driven by the natural-language question.
    """
    q = (question or "").strip()
    t0 = time.perf_counter()
    if retriever == "symptoms":
        res = kg.retrieve_by_symptoms([q]) if q else {"diseases": []}
        items = res.get("diseases", [])
    elif retriever == "disease_name":
        res = kg.retrieve_disease_info(q) if q else {"diseases": []}
        items = res.get("diseases", [])
    elif retriever == "medicine_name":
        res = vec.retrieve_by_medicine_name(name) if name else {"medicines": []}
        items = res.get("medicines", [])
    else:  # medicine_search -> semantic over the question
        res = vec.retrieve_medicines(q) if q else {"medicines": []}
        items = res.get("medicines", [])
    return items, (time.perf_counter() - t0) * 1000


class RetrievalRecorder:
    """Monkey-patch the agents' retrieve_* methods to record each tool call's results."""

    SPECS = [
        ("retrieve_by_symptoms", "find_diseases_by_symptoms", "kg", "diseases"),
        ("retrieve_disease_info", "get_disease_details", "kg", "diseases"),
        ("retrieve_by_medicine_name", "get_medicine_details", "vector", "medicines"),
        ("retrieve_medicines", "search_medicines", "vector", "medicines"),
    ]

    def __init__(self, kg_agent, vector_agent):
        """Wrap every retrieve_* method on the two retrieval agents."""
        self.records = []
        for method, tool, store, key in self.SPECS:
            obj = kg_agent if store == "kg" else vector_agent
            self._wrap(obj, method, tool, store, key)

    def _wrap(self, obj, method, tool, store, key):
        """Replace one method with a timing+recording wrapper."""
        orig = getattr(obj, method)

        def wrapped(*a, **k):
            t0 = time.perf_counter()
            res = orig(*a, **k)
            dt = (time.perf_counter() - t0) * 1000
            items = [x for x in (res.get(key) or []) if isinstance(x, dict)]
            self.records.append({"tool": tool, "store": store, "latency_ms": dt,
                                 "items": items, "ranked": [x.get("name") for x in items]})
            return res

        setattr(obj, method, wrapped)

    def reset(self):
        """Clear the recorded calls before the next question."""
        self.records = []


def judge(client, question, ground_truth, answer):
    """Grade an answer against the reference with an LLM judge -> (score, verdict)."""
    prompt = (f"Question: {question}\n\nReference answer: {ground_truth}\n\n"
              f"Model answer: {answer}\n\n"
              "Does the model answer correctly cover the reference answer? "
              'Reply JSON {"verdict": "correct" | "partial" | "incorrect"}.')
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": "You are a strict grader. Output only JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0, max_tokens=40,
        )
        verdict = json.loads(r.choices[0].message.content).get("verdict", "").lower()
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(verdict, 0.0), verdict
    except Exception as e:
        return None, f"error: {str(e)[:60]}"


def main():
    """Run every benchmark question, score it, stream results, and write eval.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "agent"], default="retrieval",
                    help="retrieval = drive retrievers directly (no Groq); agent = full LLM agent")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N questions")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge answer grading")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(BENCHMARK, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading agents... ({len(rows)} questions, mode={args.mode})")
    kg = KnowledgeGraphRetrievalAgent()
    vec = VectorRetrievalAgent()

    agent = recorder = judge_client = None
    if args.mode == "agent":
        agent = MedicalAgent(kg, vec, verbose=False)
        recorder = RetrievalRecorder(kg, vec)
        judge_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if not args.no_judge else None

    print("Warming up models...")
    kg.retrieve_by_symptoms(["fever"])
    vec.retrieve_by_medicine_name("paracetamol")
    if agent:
        agent.reset()
        agent.chat("what are the side effects of paracetamol")
        agent.reset()

    per_query = []
    partial_f = open(PARTIAL_JSONL, "w", encoding="utf-8")
    print(f"Streaming per-query results to {PARTIAL_JSONL.name}")
    for i, row in enumerate(rows, 1):
        store = row.get("store", "")
        retriever = row.get("retriever", "")
        golds = golds_for_row(row)
        answer_items = answer_items_for_row(row)
        question = row["question"]
        score_mode = row.get("score_mode", "rank")
        rel_field = row.get("relevance_field", "")
        rel_terms = relevance_terms_for_row(row)
        lookup_name = golds[0] if golds else row.get("source", "")

        def rank_items(items):
            # Relevance-scored questions check a field; the rest rank against the gold set.
            if score_mode == "relevance":
                return relevance_rank(items, rel_field, rel_terms)
            return rank_of(golds, [it.get("name") for it in items])

        if args.mode == "agent":
            recorder.reset()
            agent.reset()
            t0 = time.perf_counter()
            try:
                answer = agent.chat(question)
            except Exception as e:
                answer = f"(agent error: {e})"
            e2e_ms = (time.perf_counter() - t0) * 1000
            recs = list(recorder.records)
            retr_ms = sum(r["latency_ms"] for r in recs)
            top5 = (recs and [r["ranked"][:5] for r in recs if r["store"] == store][:1] or [[]])[0]
            ranks = [rk for rk in (rank_items(r["items"]) for r in recs if r["store"] == store) if rk]
            rank = min(ranks) if ranks else None
            called = set(r["tool"] for r in recs)
            retrieval_trace = [{
                "tool": r["tool"], "store": r["store"],
                "latency_ms": round(r["latency_ms"], 1),
                "top5": r["ranked"][:5],
                "gold_rank": rank_items(r["items"]),
            } for r in recs]
        else:
            items, retr_ms = retrieve_direct(kg, vec, retriever, question, lookup_name)
            answer, e2e_ms = "", retr_ms
            ranked = [it.get("name") for it in items]
            top5 = ranked[:5]
            rank = rank_items(items)
            called = set()
            retrieval_trace = [{
                "tool": retriever, "store": store,
                "latency_ms": round(retr_ms, 1),
                "top5": ranked[:5],
                "gold_rank": rank,
            }]

        rr = 1.0 / rank if rank else 0.0
        ndcg = (1.0 / math.log2(rank + 1)) if rank and rank <= 5 else 0.0

        expected = set(t.strip() for t in row["expected_tools"].split("+"))
        routing_recall = (len(expected & called) / len(expected)) if (args.mode == "agent" and expected) else None

        if args.mode == "agent":
            ans_l = answer.lower()
            hits = sum(1 for it in answer_items if norm(it) and norm(it) in ans_l)
            evidence_recall = round(hits / len(answer_items), 3) if answer_items else 0.0
            gold_mention = 1.0 if any(same_item(g, answer) or norm(g) in ans_l for g in golds) else 0.0
            judge_score, verdict = (judge(judge_client, question, row["ground_truth"], answer)
                                    if judge_client else (None, "skipped"))
        else:
            evidence_recall = gold_mention = judge_score = None
            verdict = "n/a"

        record = {
            "id": int(row["id"]), "category": row["category"], "subtype": row["subtype"],
            "difficulty": row.get("difficulty", ""), "retrieval_type": row.get("retrieval_type", ""),
            "question": question, "gold": golds, "store": store,
            "score_mode": score_mode, "relevance_terms": rel_terms,
            "ground_truth": row.get("ground_truth", ""),
            "answer": answer,
            "retrieved_top5": top5, "rank": rank, "gold_found": rank is not None,
            "retrieval_trace": retrieval_trace,
            "recall@1": int(bool(rank and rank <= 1)),
            "recall@3": int(bool(rank and rank <= 3)), "recall@5": int(bool(rank and rank <= 5)),
            "reciprocal_rank": round(rr, 3), "ndcg@5": round(ndcg, 3),
            "tools_called": sorted(called), "expected_tools": sorted(expected),
            "routing_recall": routing_recall,
            "evidence_recall": evidence_recall, "gold_mention": gold_mention,
            "judge_score": judge_score, "judge_verdict": verdict,
            "latency_ms": {"end_to_end": round(e2e_ms, 1), "retrieval": round(retr_ms, 1)},
        }
        per_query.append(record)
        # Flush + fsync so a crash keeps this row on disk.
        partial_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        partial_f.flush()
        os.fsync(partial_f.fileno())
        print(f"  [{i}/{len(rows)}] {row['category']:15s} rank={rank} "
              f"rr={rr:.2f} retr={retr_ms:.0f}ms" + (f" judge={verdict}" if args.mode == "agent" else ""))

    partial_f.close()
    report = aggregate(per_query, rows, args)
    OUT_JSON.write_text(json.dumps({"config": report["config"], "global": report["global"],
                                    "by_category": report["by_category"],
                                    "by_retrieval_type": report["by_retrieval_type"],
                                    "per_query": per_query},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)


def _avg(xs):
    """Mean of the non-None values, or 0.0 if there are none."""
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def aggregate(pq, rows, args):
    """Build the global, by-category, and by-retrieval-type metric report."""
    agent_mode = args.mode == "agent"

    def block(items):
        """Compute the metric block for one group of per-query records."""
        b = {
            "n": len(items),
            "recall@1": _avg([x["recall@1"] for x in items]),
            "recall@3": _avg([x["recall@3"] for x in items]),
            "recall@5": _avg([x["recall@5"] for x in items]),
            "mrr": _avg([x["reciprocal_rank"] for x in items]),
            "ndcg@5": _avg([x["ndcg@5"] for x in items]),
            "latency_ms": {"retrieval": stats([x["latency_ms"]["retrieval"] for x in items])},
        }
        if agent_mode:
            b["routing_recall"] = _avg([x["routing_recall"] for x in items])
            b["evidence_recall"] = _avg([x["evidence_recall"] for x in items])
            b["gold_mention_rate"] = _avg([x["gold_mention"] for x in items])
            b["judge_score"] = _avg([x["judge_score"] for x in items])
            b["latency_ms"]["end_to_end"] = stats([x["latency_ms"]["end_to_end"] for x in items])
        return b

    by_cat = {}
    for cat in sorted(set(x["category"] for x in pq)):
        by_cat[cat] = block([x for x in pq if x["category"] == cat])

    # known_item (exact lookup) vs semantic vs multi_hop -- the honesty slice.
    by_rtype = {}
    for rt in sorted(set(x.get("retrieval_type", "") for x in pq if x.get("retrieval_type"))):
        by_rtype[rt] = block([x for x in pq if x.get("retrieval_type") == rt])

    return {
        "config": {
            "mode": args.mode,
            "agent_model": "llama-3.3-70b-versatile" if agent_mode else None,
            "judge_model": (None if (args.no_judge or not agent_mode) else JUDGE_MODEL),
            "benchmark": str(BENCHMARK.name), "n_questions": len(pq),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "global": block(pq),
        "by_category": by_cat,
        "by_retrieval_type": by_rtype,
    }


def print_summary(report):
    """Print the global, by-retrieval-type, and by-category results to the console."""
    g = report["global"]
    agent_mode = report["config"]["mode"] == "agent"
    print("\n" + "=" * 60)
    print(f"GLOBAL RESULTS  (mode={report['config']['mode']})")
    print("=" * 60)
    print(f"  Questions         : {g['n']}")
    print(f"  Recall@1 / @3 / @5: {g['recall@1']:.2%} / {g['recall@3']:.2%} / {g['recall@5']:.2%}")
    print(f"  MRR               : {g['mrr']:.3f}")
    print(f"  nDCG@5            : {g['ndcg@5']:.3f}")
    print(f"  Latency retrieval : {g['latency_ms']['retrieval']['mean']:.1f} ms "
          f"(p50 {g['latency_ms']['retrieval']['p50']:.0f}, p95 {g['latency_ms']['retrieval']['p95']:.0f})")
    if agent_mode:
        print(f"  Routing recall    : {g['routing_recall']:.2%}")
        print(f"  Evidence recall   : {g['evidence_recall']:.2%}")
        print(f"  Gold mention rate : {g['gold_mention_rate']:.2%}")
        print(f"  Judge score       : {g['judge_score']:.3f}")
        print(f"  Latency end-to-end: {g['latency_ms']['end_to_end']['mean']:.0f} ms "
              f"(p95 {g['latency_ms']['end_to_end']['p95']:.0f})")
    print("\n  By retrieval_type:")
    for rt, b in report.get("by_retrieval_type", {}).items():
        print(f"    {rt:16s} n={b['n']:<3d} R@1={b['recall@1']:.2%}  R@5={b['recall@5']:.2%}  "
              f"MRR={b['mrr']:.2f}  nDCG@5={b['ndcg@5']:.2f}  retr={b['latency_ms']['retrieval']['mean']:.1f}ms")
    print("\n  By category:")
    for cat, b in report["by_category"].items():
        extra = f"  judge={b['judge_score']:.2f}" if agent_mode else ""
        print(f"    {cat:16s} R@5={b['recall@5']:.2%}  MRR={b['mrr']:.2f}  "
              f"nDCG@5={b['ndcg@5']:.2f}{extra}  retr={b['latency_ms']['retrieval']['mean']:.1f}ms")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
