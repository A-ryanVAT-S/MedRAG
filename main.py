# CLI entry point. The central agent does the reasoning; KG and Vector are its tools.
import sys
from dotenv import load_dotenv

# Keep stdout/stderr UTF-8 so a stray character (e.g. Rs symbol) can't crash a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

from agents.agent2_kg_retrieval import KnowledgeGraphRetrievalAgent
from agents.agent3_vector_retrieval import VectorRetrievalAgent
from agents.agent_core import MedicalAgent


class MedRAG:
    """Wires the KG and vector retrieval agents into the central medical agent."""

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None, chroma_path=None):
        """Initialize the retrieval agents and the reasoning agent."""
        print("\n" + "*"*80)
        print("Initializing MedRAG")
        print("*"*80)

        self.kg_agent = KnowledgeGraphRetrievalAgent(
            neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password
        )
        self.vector_agent = VectorRetrievalAgent(chroma_path=chroma_path)
        self.agent = MedicalAgent(self.kg_agent, self.vector_agent)

        print("\nReady!\n")

    def ask(self, query: str) -> str:
        """Answer a single query through the agent."""
        return self.agent.chat(query)

    def reset(self):
        """Start a new conversation."""
        self.agent.reset()


def run_interactive_mode(med: MedRAG):
    """Run the interactive REPL loop until the user exits."""
    print("\n" + "*"*80)
    print(" Interactive Mode - 'exit' to quit, 'reset' to start a new conversation")
    print("*"*80 + "\n")

    while True:
        try:
            query = input(" You: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n Thank you for using MedRAG. Stay healthy!")
                break
            if query.lower() in ['reset', 'new', 'clear']:
                med.reset()
                print(" [conversation reset]\n")
                continue
            if not query:
                continue

            answer = med.ask(query)
            print("\n" + "-"*80)
            print(answer)
            print("-"*80 + "\n")

        except KeyboardInterrupt:
            print("\n\n Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again.\n")


def main():
    """Entry point: initialize MedRAG and start interactive mode."""
    try:
        med = MedRAG()
        run_interactive_mode(med)
    except Exception as e:
        print(f" Failed to initialize MedRAG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
