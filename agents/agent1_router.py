#Query Agent
import os
import json
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parent.parent / ".env")

# Agent 1: Query Routing Agent
class QueryRoutingAgent:
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"
        print(f" [Agent 1] Connected to Groq LLM (Model: {self.model_name})")
        
        self.system_prompt = """Extract medical entities from queries and return JSON.

**Entity Extraction (MOST IMPORTANT):**
1. symptoms: ANY health problems/feelings mentioned → list them
2. disease: ANY disease/condition NAME mentioned → extract it
3. medicine: ANY medicine NAME  mentioned → extract it

**Key rule: If user mentions ANYTHING medical (diabetes, cancer, flu, etc.), extract it as disease if not related with medicine!**

**Intent (choose ONE):**
- symptom_to_disease: symptoms listed, asking what disease
- disease_info: disease name mentioned
- medicine_info: medicine name mentioned
- mixed: multiple things

**Retrieval:**
- If disease mentioned → use_kg=true, use_vector=true (always both!)
- If only medicine → use_kg=false, use_vector=true
- If only symptoms → use_kg=true, use_vector=false

**Examples:**
Q: "What medicines treat diabetes and its side effects?"
→ {"intent": "disease_info", "entities": {"symptoms": [], "disease": "diabetes", "medicine": null}, "use_kg": true, "use_vector": true}(vector as we need diabetes medicine side effects)

Q: "I have fever"
→ {"intent": "symptom_to_disease", "entities": {"symptoms": ["fever"], "disease": null, "medicine": null}, "use_kg": true, "use_vector": false}

Return ONLY JSON, no text:
{"intent": "disease_info", "entities": {"symptoms": [], "disease": "diabetes", "medicine": null}, "use_kg": true, "use_vector": true}"""
    
    # Analyze query
    def route_query(self, query: str) -> Dict:
    
        print("\n [Agent 1] Analyzing query intent...")
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                model=self.model_name,
                temperature=0.1,
                max_tokens=500
            )
            response = chat_completion.choices[0].message.content
            
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Extract JSON object
            start = response.find("{")
            end = response.rfind("}") + 1

            if start == -1 or end <= start:
                raise ValueError(f"No valid JSON object found in response")

            json_str = response[start:end]
            
            routing_data = json.loads(json_str)

            
            if "intent" not in routing_data or "entities" not in routing_data:
                raise ValueError("Invalid routing structure")
            
            valid_intents = {
                "symptom_to_disease",
                "disease_info",
                "medicine_info",
                "mixed"
            }
            
            if routing_data["intent"] not in valid_intents:
                raise ValueError("Invalid intent value")
            
            routing_data = self.enforce_retrieval_policy(routing_data)
            
            print(f"   Intent: {routing_data['intent']}")
            print(f"   Entities: {routing_data['entities']}")
            print(f"   LLM Routing Decision: KG={routing_data['use_kg']}, Vector={routing_data['use_vector']}")
            
            return routing_data
            
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            print(f"   LLM JSON failed: {str(e)[:80]}")
            try:
                print(f"   DEBUG Raw response: >>>{response[:200]}<<<")
                print(f"   DEBUG json_str: >>>{json_str[:200] if 'json_str' in dir() else 'N/A'}<<<")
            except:
                pass
            
            # Fallback: Use smart defaults based on query content
            fallback = self._get_smart_fallback(query)
            print(f"   Using fallback routing: {fallback['intent']}, KG={fallback['use_kg']}, Vector={fallback['use_vector']}")
            return fallback
        except Exception as e:
            print(f"   Unexpected error: {type(e).__name__}: {e}")
            # Final fallback
            fallback = self._get_smart_fallback(query)
            print(f"   Using emergency fallback: {fallback['intent']}")
            return fallback
    
    def _get_smart_fallback(self, query: str) -> Dict:
        """Simple fallback - just detect if query is about symptoms, disease, or medicine."""
        query_lower = query.lower()
        
        # Simple keyword detection (no exhaustive lists)
        symptom_indicators = ["pain", "ache", "feel", "experiencing", "symptom", "have"]
        has_symptom_language = any(word in query_lower for word in symptom_indicators)
        
        medicine_indicators = ["medicine", "medication", "drug", "pill", "side effect", "dosage"]
        asks_medicine = any(word in query_lower for word in medicine_indicators)
        
        disease_indicators = ["disease", "condition", "what could", "associated", "diagnosis"]
        asks_disease = any(word in query_lower for word in disease_indicators)
        
        print(f"   [Fallback] Symptom language: {has_symptom_language}, Asks disease: {asks_disease}, Asks medicine: {asks_medicine}")
        
        # Simple routing logic
        if has_symptom_language:
            return {
                "intent": "symptom_to_disease",
                "entities": {"symptoms": [], "disease": None, "medicine": None},
                "use_kg": True,
                "use_vector": asks_medicine
            }
        elif asks_medicine:
            return {
                "intent": "medicine_info",
                "entities": {"symptoms": [], "disease": None, "medicine": None},
                "use_kg": False,
                "use_vector": True
            }
        elif asks_disease:
            return {
                "intent": "disease_info",
                "entities": {"symptoms": [], "disease": None, "medicine": None},
                "use_kg": True,
                "use_vector": True
            }
        else:
            return {
                "intent": "mixed",
                "entities": {"symptoms": [], "disease": None, "medicine": None},
                "use_kg": True,
                "use_vector": True
            }
    
    def enforce_retrieval_policy(self, routing_data: Dict) -> Dict:
        intent = routing_data.get("intent")
        entities = routing_data.get("entities", {})
        
        if intent == "medicine_info":
            routing_data["use_kg"] = False
            routing_data["use_vector"] = True
            entities["symptoms"] = []
            entities["disease"] = None
        
        elif intent == "symptom_to_disease":
            routing_data["use_kg"] = True
            routing_data["use_vector"] = False
            entities["medicine"] = None
        
        elif intent in ("disease_info", "mixed"):
            routing_data["use_kg"] = True
            routing_data["use_vector"] = True
        
        routing_data["entities"] = entities
        return routing_data
    
    # checking knowledge graph need
    def should_use_kg(self, routing_data: Dict) -> bool:
        return routing_data.get("use_kg", False)
    
    # checking vector db need
    def should_use_vector(self, routing_data: Dict) -> bool:
        return routing_data.get("use_vector", False)


if __name__ == "__main__":
    agent = QueryRoutingAgent()
    
    test_queries = [
        "I have fever and cough, what could it be?",
        "What is influenza?",
        "Tell me about paracetamol side effects",
        "I have a headache, should I take aspirin?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = agent.route_query(query)
        print(f"KG Needed: {agent.should_use_kg(result)}")
        print(f"Vector Needed: {agent.should_use_vector(result)}")
