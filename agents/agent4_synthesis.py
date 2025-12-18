#Response Synthesis Agent

import json
import os
from typing import Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ResponseSynthesisAgent:
    
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"
        print(f" [Agent 4] Connected to Groq LLM (Model: {self.model})")
        
        self.synthesis_prompt = """You are a medical information assistant.
Your task is to synthesize ONLY the retrieved data provided below.

User Query:
{{query}}

-----------------------------------
Knowledge Graph Data (may be EMPTY):
{{kg_data}}

Vector Database Data (may be EMPTY):
{{vector_data}}

-----------------------------------
STRICT RULES:
1. Use ONLY the data provided above
2. If Knowledge Graph data is empty or irrelevant → DO NOT mention diseases
3. If Vector data is empty → DO NOT mention medicines
4. DO NOT invent links between diseases and medicines
5. DO NOT add symptoms, diseases, or medicines not present in the data
6. DO NOT give medical advice — informational only

-----------------------------------
Response Structure:
- If KG data exists:
  • List diseases with matched symptoms
  • Mention treatments exactly as provided
- If Vector data exists:
  • List medicines with descriptions
  • Mention side effects or interactions ONLY if present in data
- If only one source exists, ignore the other completely

End with:
"Educational use only. Not medical advice."

-----------------------------------
Generate a clear, structured response:
"""
    
    def synthesize_response(
        self,
        user_query: str,
        kg_data: Optional[Dict] = None,
        vector_data: Optional[Dict] = None
    ) -> Dict:
        
        print("\n [Agent 4] Synthesizing response...")
        
        # Format KG data
        kg_text = self._format_kg_data(kg_data) if kg_data else "No knowledge graph data available."
        
        # Format vector data
        vector_text = self._format_vector_data(vector_data) if vector_data else "No vector database data available."
        
        # Generate synthesis
        try:
            prompt = self.synthesis_prompt.replace("{{query}}", user_query).replace("{{kg_data}}", kg_text).replace("{{vector_data}}", vector_text)
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical information assistant that synthesizes retrieved data accurately."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1024
            )
            response_text = chat_completion.choices[0].message.content
            
            # Build structured response
            response = {
                "answer": response_text.strip(),
                "citations": self._extract_citations(kg_data, vector_data),
                "sources_used": self._get_sources_summary(kg_data, vector_data),
                "disclaimer": self._get_disclaimer()
            }
            
            print(" Response generated successfully")
            return response
            
        except Exception as e:
            print(f" Error generating response: {e}")
            return {
                "answer": "I apologize, but I encountered an error while generating the response.",
                "citations": [],
                "sources_used": {"kg": False, "vector": False},
                "disclaimer": self._get_disclaimer(),
                "error": str(e)
            }
    
    def _format_kg_data(self, kg_data: Dict) -> str:
        # Format knowledge graph data for prompt.

        if not kg_data or not kg_data.get("diseases"):
            return "No diseases found in knowledge graph."
        
        formatted = "Knowledge Graph Retrieved:\n"
        for disease in kg_data.get("diseases", []):
            formatted += f"\n[Disease: {disease.get('name', 'Unknown')}]\n"
            
            if disease.get('description'):
                formatted += f"  Description: {disease['description']}\n"
            
            if disease.get('symptoms'):
                formatted += f"  Symptoms: {', '.join(disease['symptoms'])}\n"
            
            if disease.get('treatments'):
                formatted += f"  Treatments: {disease['treatments']}\n"
            
            if disease.get('matched_symptoms'):
                formatted += f"  Matched {disease['matched_symptoms']} symptoms\n"
        
        return formatted
    
    def _format_vector_data(self, vector_data: Dict) -> str:
        """Format vector database data for prompt."""
        if not vector_data or not vector_data.get("medicines"):
            return "No medicines found in vector database."
        
        formatted = "Vector Database Retrieved:\n"
        for medicine in vector_data.get("medicines", []):
            formatted += f"\n[Medicine: {medicine.get('name', 'Unknown')}]\n"
            formatted += f"  Summary: {medicine.get('summary', 'No summary')}\n"
            
            if medicine.get('side_effects') and medicine['side_effects'] != "Not available":
                formatted += f"  Side Effects: {medicine['side_effects']}\n"
            
            if medicine.get('dosage') and medicine['dosage'] != "Consult healthcare provider":
                formatted += f"  Dosage: {medicine['dosage']}\n"
            
            if medicine.get('interactions') and medicine['interactions'] != "Not available":
                formatted += f"  Interactions: {medicine['interactions']}\n"
        
        return formatted
    
    def _extract_citations(
        self,
        kg_data: Optional[Dict],
        vector_data: Optional[Dict]
    ) -> List[Dict]:
        """Extract structured citations from retrieved data."""
        citations = []
        
        # Add KG citations
        if kg_data and kg_data.get("diseases"):
            for disease in kg_data["diseases"]:
                citations.append({
                    "source": "Knowledge Graph",
                    "type": "Disease Information",
                    "entity": disease.get("name", "Unknown"),
                    "data_type": kg_data.get("query_type", "unknown")
                })
        
        # Add vector citations
        if vector_data and vector_data.get("medicines"):
            for medicine in vector_data["medicines"]:
                citations.append({
                    "source": "Vector Database",
                    "type": "Medicine Information",
                    "entity": medicine.get("name", "Unknown"),
                    "similarity_score": medicine.get("similarity_score")
                })
        
        return citations
    
    def _get_sources_summary(
        self,
        kg_data: Optional[Dict],
        vector_data: Optional[Dict]
    ) -> Dict:
        """Summarize which sources were used."""
        return {
            "knowledge_graph": bool(kg_data and kg_data.get("diseases")),
            "vector_database": bool(vector_data and vector_data.get("medicines")),
            "kg_results_count": len(kg_data.get("diseases", [])) if kg_data else 0,
            "vector_results_count": len(vector_data.get("medicines", [])) if vector_data else 0
        }
    
    def _get_disclaimer(self) -> str:
        # Return medical disclaimer.
        return (
            " MEDICAL DISCLAIMER: This information is for educational purposes only "
            "and should not replace professional medical advice. Always consult with a "
            "qualified healthcare provider for diagnosis and treatment decisions."
        )
    
    def format_for_display(self, response: Dict) -> str:
        # Format response for beautiful CLI display.
       
        output = []
        output.append("\n" + "="*80)
        output.append(" MEDICAL QUERY RESPONSE")
        output.append("="*80 + "\n")
        
        # Main answer
        output.append(" Answer:")
        output.append("-" * 80)
        output.append(response.get("answer", "No answer available."))
        output.append("")
        
        # Sources used
        sources = response.get("sources_used", {})
        output.append("\n Sources Used:")
        output.append("-" * 80)
        if sources.get("knowledge_graph"):
            output.append(f"   Knowledge Graph: {sources.get('kg_results_count', 0)} results")
        if sources.get("vector_database"):
            output.append(f"   Vector Database: {sources.get('vector_results_count', 0)} results")
        if not sources.get("knowledge_graph") and not sources.get("vector_database"):
            output.append("   No sources available")
        
        # Citations
        citations = response.get("citations", [])
        if citations:
            output.append("\n Citations:")
            output.append("-" * 80)
            for i, citation in enumerate(citations, 1):
                output.append(f"  [{i}] {citation['source']}")
                output.append(f"      Type: {citation['type']}")
                output.append(f"      Entity: {citation['entity']}")
                if citation.get('similarity_score'):
                    output.append(f"      Similarity: {citation['similarity_score']:.3f}")
        
        # Disclaimer
        output.append("\n Disclaimer:")
        output.append("-" * 80)
        output.append(response.get("disclaimer", ""))
        output.append("\n" + "="*80 + "\n")
        
        return "\n".join(output)


if __name__ == "__main__":
    # Test the agent
    agent = ResponseSynthesisAgent()
    
    # Mock data for testing
    mock_kg = {
        "diseases": [
            {
                "name": "Influenza",
                "description": "Viral respiratory infection",
                "symptoms": ["fever", "cough", "fatigue"],
                "treatments": "Rest, fluids, antivirals"
            }
        ],
        "query_type": "symptom_mapping"
    }
    
    mock_vector = {
        "medicines": [
            {
                "name": "Paracetamol",
                "summary": "Used to reduce fever and relieve pain",
                "side_effects": "Rare: nausea, liver toxicity at high doses",
                "dosage": "500-1000mg every 4-6 hours"
            }
        ]
    }
    
    response = agent.synthesize_response(
        "I have fever and cough, what should I do?",
        kg_data=mock_kg,
        vector_data=mock_vector
    )
    
    print(agent.format_for_display(response))
