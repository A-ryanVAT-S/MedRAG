#MedRAG: Medical Retrieval-Augmented Generation System

import sys
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from agents.agent1_router import QueryRoutingAgent
from agents.agent2_kg_retrieval import KnowledgeGraphRetrievalAgent
from agents.agent3_vector_retrieval import VectorRetrievalAgent
from agents.agent4_synthesis import ResponseSynthesisAgent


class MedRAGOrchestrator:
    # Initialize all agents
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        chroma_path: str = None
    ):
        print("\n" + "*"*80)
        print("Initializing MedRAG Multi-Agent System")
        print("*"*80)
        
        self.router = QueryRoutingAgent()  # Uses Groq API from .env
        self.kg_agent = KnowledgeGraphRetrievalAgent(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        self.vector_agent = VectorRetrievalAgent(chroma_path=chroma_path)
        self.synthesis_agent = ResponseSynthesisAgent()
        
        print("\nAll agents initialized successfully!\n")
    
    def process_query(self, query: str, verbose: bool = True) -> Dict:
       # Orchestrate the multi-agent retrieval and synthesis process
        if verbose:
            print("\n" + "*"*80)
            print(f"Processing Query: {query}")
            print("*"*80)
        
        # Step 1: Route query and extract entities
        routing_data = self.router.route_query(query)
        
        # Step 2: Determine retrieval strategy
        use_kg = self.router.should_use_kg(routing_data)
        use_vector = self.router.should_use_vector(routing_data)
        
        if verbose:
            print(f"\n Retrieval Strategy:")
            print(f"   Knowledge Graph: {'Needed' if use_kg else 'Not Needed'}")
            print(f"   Vector Database: {'Needed' if use_vector else 'Not Needed'}")
        
        # Step 3: Execute retrievals
        kg_data = None
        vector_data = None
        
        if use_kg:
            kg_data = self._execute_kg_retrieval(routing_data)
        
        if use_vector:
            vector_data = self._execute_vector_retrieval(routing_data, kg_data)
        
        # Step 4: Synthesize response
        response = self.synthesis_agent.synthesize_response(
            user_query=query,
            kg_data=kg_data,
            vector_data=vector_data
        )
        
        # Add routing metadata
        response["routing"] = routing_data
        response["retrieval_strategy"] = {
            "kg_used": use_kg,
            "vector_used": use_vector
        }
        
        return response
    
    def _execute_kg_retrieval(self, routing_data: Dict) -> Optional[Dict]:
        # Execute knowledge graph retrieval based on routing.

        entities = routing_data.get("entities", {})
        intent = routing_data.get("intent", "")
        
        # Symptom-based retrieval
        if entities.get("symptoms") and len(entities["symptoms"]) > 0:
            return self.kg_agent.retrieve_by_symptoms(entities["symptoms"])
        
        # Disease-based retrieval
        elif entities.get("disease"):
            return self.kg_agent.retrieve_disease_info(entities["disease"])
        
        # Generic disease info query
        elif intent == "disease_info":
            return self.kg_agent.retrieve_disease_info("general")
        
        return None
    
    def _execute_vector_retrieval(
        self,
        routing_data: Dict,
        kg_data: Optional[Dict]
    ) -> Optional[Dict]:
        """Execute vector database retrieval based on routing."""
        entities = routing_data.get("entities", {})
        intent = routing_data.get("intent", "")
        
        # Build context from KG if available
        disease_context = None
        if kg_data and kg_data.get("diseases"):
            disease_names = [d["name"] for d in kg_data["diseases"][:3]]  # Top 3 diseases
            disease_context = ", ".join(disease_names)
        
        # Priority 1: Medicine-specific retrieval
        if entities.get("medicine"):
            return self.vector_agent.retrieve_by_medicine_name(entities["medicine"])
        
        # Priority 2: Disease-based medicine retrieval (if disease mentioned)
        if entities.get("disease"):
            return self.vector_agent.retrieve_by_disease(entities["disease"])
        
        # Priority 3: Use KG disease context for medicine retrieval (most common case)
        if disease_context:
            return self.vector_agent.retrieve_by_disease(disease_context)
        
        # Priority 4: Symptom-based medicine retrieval (fallback)
        if entities.get("symptoms") and len(entities["symptoms"]) > 0:
            query = f"medicines for treatment of {', '.join(entities['symptoms'])}"
            return self.vector_agent.retrieve_medicines(query)
        
        # Priority 5: General fallback if vector DB is needed but no clear path
        if intent in ["mixed", "disease_info"]:
            return self.vector_agent.retrieve_medicines("general treatment options")
        
        return None


def run_interactive_mode(orchestrator: MedRAGOrchestrator):
    """Run interactive CLI mode for continuous queries."""
    print("\n" + "*"*80)
    print(" Interactive Mode - Type 'exit' or 'quit' to end session")
    print("*"*80 + "\n")
    
    while True:
        try:
            # Get user input
            query = input(" Enter your medical query: ").strip()
            
            # Check for exit
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n Thank you for using MedRAG. Stay healthy!")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            # Process query
            response = orchestrator.process_query(query, verbose=True)
            
            # Display formatted response
            formatted_output = orchestrator.synthesis_agent.format_for_display(response)
            print(formatted_output)
            
        except KeyboardInterrupt:
            print("\n\n Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again with a different query.\n")

def main():
    try:
        orchestrator = MedRAGOrchestrator()
        run_interactive_mode(orchestrator)
    except Exception as e:
        print(f" Failed to initialize MedRAG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
