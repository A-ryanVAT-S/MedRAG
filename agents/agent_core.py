# Tool-calling agent that plans tool use, combines results, and stays within the data.
import os
import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from groq import BadRequestError

load_dotenv(Path(__file__).parent.parent / ".env")


SYSTEM_PROMPT = """You are MedRAG, an educational medical information assistant.
You answer ONLY using data returned by your tools. You never use outside medical
knowledge and you never invent facts.

# YOUR DATA (this is ALL you have)

1. KNOWLEDGE GRAPH of diseases. Tools: find_diseases_by_symptoms, get_disease_details.
   Per disease it stores ONLY:
     - name
     - symptoms (list)
     - treatments (list of general treatment names)
     - contagious (true/false)
     - chronic (true/false)
   It does NOT store: causes/etiology, prevention, prognosis, recovery time, mortality,
   severity, risk factors, or diagnostic tests.

2. MEDICINE DATABASE (~195,000 medicines). Tools: get_medicine_details, search_medicines.
   Per medicine it stores ONLY:
     - product_name
     - salt_composition (the active ingredient)
     - sub_category
     - price
     - manufacturer
     - description (what the medicine is used for, free text)
     - side_effects (list)
     - drug_interactions (list of other drugs it interacts with)
   It does NOT store: dosage/how much to take, pregnancy or age-specific safety (unless
   it happens to be written in the description), efficacy ranking, or the ability to
   sort/filter by price or manufacturer.

# TOOLS

- find_diseases_by_symptoms(symptoms): given one or more symptoms, returns candidate
  diseases WITH each disease's full symptom list, treatments, contagious, chronic.
  Use for "what could these symptoms be" and as the first step before suggesting medicines.

- get_disease_details(disease_name): full record for a single named disease.
  Use for "symptoms of X", "how is X treated", "is X contagious/chronic".

- get_medicine_details(medicine_name): full record for a named medicine. Use for side
  effects, composition/ingredient, price, manufacturer, drug interactions, or to check
  whether a medicine's description covers a condition ("will X help my Y").

- search_medicines(query): semantic search over medicine descriptions AND side effects.
  Use for "medicines for <condition>" and "which medicines cause <side effect>".
  Do NOT use it for a named medicine's composition, price, manufacturer, or interactions
  (those live only in the full record) - call get_medicine_details for a named medicine.

# HOW TO REASON

- Pick the tool(s) that answer the question. You may call several tools, in sequence.
  Example: symptoms -> find_diseases_by_symptoms -> then search_medicines for the most
  likely disease to suggest treatment medicines.
- DIFFERENTIAL: if several diseases match the symptoms AND the user wants a likely
  diagnosis or medicines, FIRST ask ONE short follow-up question. List 4-8 DISTINGUISHING
  symptoms (taken from the tool results) and ask which ones the user also has, so you can
  narrow down. Only after they answer should you commit to the most likely disease.
- "Will medicine X help my Y?": call get_medicine_details(X); decide from its description
  whether it covers Y (optionally call find_diseases_by_symptoms to see what Y relates to).
  State clearly whether the data supports it or not.
- Drug interactions / "can I take X with Y": get_medicine_details and read drug_interactions.

# HONESTY (CRITICAL)

- Use ONLY tool results. Answer the parts you can.
- If the user asks for something not in the data (causes, prevention, dosage, prognosis,
  age/pregnancy safety, cheapest option, etc.), say plainly: "Our dataset does not include
  <that>." Then give the related information you DO have.
- If a tool returns nothing, say you couldn't find it in the database. Do not guess.
- Never fabricate diseases, medicines, dosages, or disease-medicine links not in the results.
- Cite the disease/medicine names you used.
- End EVERY answer with: "Educational use only - not medical advice. Consult a doctor."
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_diseases_by_symptoms",
            "description": "Find candidate diseases from a list of symptoms. Returns each "
                           "disease with its full symptom list, treatments, contagious and "
                           "chronic flags. Use for diagnosis and as a first step before "
                           "recommending medicines for symptoms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Symptoms mentioned by the user, e.g. ['fever','itching']",
                    }
                },
                "required": ["symptoms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_disease_details",
            "description": "Get the full knowledge-graph record for one named disease: its "
                           "symptoms, treatments, contagious and chronic flags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {"type": "string", "description": "Disease name, e.g. 'malaria'"}
                },
                "required": ["disease_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_medicine_details",
            "description": "Get the full record for a named medicine: salt_composition, "
                           "sub_category, price, manufacturer, description (use), side_effects, "
                           "and drug_interactions. Use for side effects, ingredients, price, "
                           "manufacturer, interactions, or whether a medicine treats a condition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medicine_name": {"type": "string", "description": "Medicine name, e.g. 'cetirizine'"}
                },
                "required": ["medicine_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_medicines",
            "description": "Semantic search over medicine descriptions AND side effects. Use for "
                           "'medicines for <condition>', 'which medicines cause <side effect>', or "
                           "'medicines containing <ingredient>'. Returns matching medicine records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search, e.g. 'medicines that cause swelling'"}
                },
                "required": ["query"],
            },
        },
    },
]


class MedicalAgent:
    """Drives the Groq LLM through tool-call rounds against the KG and vector agents."""

    def __init__(self, kg_agent, vector_agent, model: str = "llama-3.3-70b-versatile",
                 max_tool_rounds: int = 6, verbose: bool = True):
        """Wire up the Groq client, retrieval agents, and a fresh message history."""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.kg_agent = kg_agent
        self.vector_agent = vector_agent
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose
        self.messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f" [Core Agent] Ready (model: {self.model})")

    def reset(self):
        """Drop conversation history, keep the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _dispatch(self, name: str, args: Dict) -> Dict:
        """Run the named tool and trim its result for the LLM context."""
        try:
            if name == "find_diseases_by_symptoms":
                syms = args.get("symptoms") or []
                if isinstance(syms, str):
                    syms = [syms]
                data = self.kg_agent.retrieve_by_symptoms(syms)
                diseases = []
                for d in data.get("diseases", [])[:5]:
                    diseases.append({
                        "name": d.get("name"),
                        "symptoms": (d.get("symptoms") or [])[:15],
                        "treatments": (d.get("treatments") or [])[:15],
                        "contagious": d.get("contagious"),
                        "chronic": d.get("chronic"),
                    })
                return {"diseases": diseases} if diseases else {"diseases": [], "note": "No matching diseases found in the knowledge graph."}

            if name == "get_disease_details":
                data = self.kg_agent.retrieve_disease_info(args.get("disease_name", ""))
                ds = data.get("diseases", [])
                if not ds:
                    return {"note": f"Disease '{args.get('disease_name')}' not found in the knowledge graph."}
                d = ds[0]
                return {
                    "name": d.get("name"),
                    "symptoms": d.get("symptoms") or [],
                    "treatments": d.get("treatments") or [],
                    "contagious": d.get("contagious"),
                    "chronic": d.get("chronic"),
                }

            if name == "get_medicine_details":
                data = self.vector_agent.retrieve_by_medicine_name(args.get("medicine_name", ""))
                return {"medicines": self._trim_medicines(data.get("medicines", []), top=3)}

            if name == "search_medicines":
                data = self.vector_agent.retrieve_medicines(args.get("query", ""))
                return {"medicines": self._trim_medicines(data.get("medicines", []), top=5)}

            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": f"Tool '{name}' failed: {e}"}

    @staticmethod
    def _trim_medicines(medicines: List[Dict], top: int) -> List[Dict]:
        """Dedup by product name, keep top N, and truncate descriptions for context."""
        out, seen = [], set()
        for m in medicines:
            key = (m.get("name") or "").lower()
            if key in seen:
                continue
            seen.add(key)
            desc = m.get("description") or ""
            out.append({
                "name": m.get("name"),
                "salt_composition": m.get("salt_composition"),
                "sub_category": m.get("sub_category"),
                "price": m.get("price"),
                "manufacturer": m.get("manufacturer"),
                "description": desc[:400] + ("..." if len(desc) > 400 else ""),
                "side_effects": m.get("side_effects") or [],
                "drug_interactions": m.get("drug_interactions") or [],
            })
            if len(out) >= top:
                break
        return out

    def chat(self, user_message: str) -> str:
        """Answer one user message, running tool-call rounds until the LLM responds."""
        self.messages.append({"role": "user", "content": user_message})

        for round_i in range(self.max_tool_rounds):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1200,
                )
            except BadRequestError as e:
                # llama on Groq sometimes emits a malformed tool call; recover and continue.
                parsed = self._parse_failed_tool(e)
                if parsed:
                    name, args = parsed
                    if self.verbose:
                        print(f"   [tool*] recovered {name}({args})")
                    call_id = f"recovered_{round_i}"
                    self.messages.append({
                        "role": "assistant", "content": "",
                        "tool_calls": [{"id": call_id, "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}],
                    })
                    result = self._dispatch(name, args)
                    self.messages.append({
                        "role": "tool", "tool_call_id": call_id, "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue
                if self.verbose:
                    print(f"   [warn] tool call failed, answering without tools: {str(e)[:120]}")
                return self._answer_without_tools()

            msg = resp.choices[0].message

            assistant_msg: Dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not msg.tool_calls:
                return msg.content or "(no response)"

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if self.verbose:
                    print(f"   [tool] {tc.function.name}({args})")
                result = self._dispatch(tc.function.name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return self._answer_without_tools()

    def _answer_without_tools(self) -> str:
        """Final answer pass with no tools (fallback or after the tool-round budget)."""
        self.messages.append({
            "role": "user",
            "content": "Now give your best answer using only the information gathered above. "
                       "Clearly state anything our dataset does not cover. End with the disclaimer.",
        })
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages, temperature=0.2, max_tokens=1200,
        )
        return resp.choices[0].message.content or "(no response)"

    @staticmethod
    def _parse_failed_tool(e: BadRequestError) -> Optional[Tuple[str, Dict]]:
        """Pull (tool_name, args) out of a Groq tool_use_failed error's failed_generation."""
        raw = ""
        try:
            body = getattr(e, "body", None)
            if isinstance(body, dict):
                raw = body.get("error", {}).get("failed_generation", "") or ""
        except Exception:
            raw = ""
        if not raw:
            raw = str(e)

        # Seen from llama on Groq: <function=NAME {"arg": "..."} </function>
        m = re.search(r"<function=(\w+)\s*(\{.*?\})\s*</?function>", raw, re.DOTALL)
        if not m:
            m = re.search(r'"name"\s*:\s*"(\w+)".*?("arguments"\s*:\s*)?(\{.*\})', raw, re.DOTALL)
            if m:
                name, args_str = m.group(1), m.group(3)
            else:
                return None
        else:
            name, args_str = m.group(1), m.group(2)

        valid = {"find_diseases_by_symptoms", "get_disease_details",
                 "get_medicine_details", "search_medicines"}
        if name not in valid:
            return None
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return None
        return name, args
