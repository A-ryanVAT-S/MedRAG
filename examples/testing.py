"""
MedRAG System Testing Script
Runs example queries and generates output log for demonstration
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import MedRAGOrchestrator


class LogCapture:
    """Captures both console output and writes to log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")


def run_test_query(orchestrator, query_num, query_text, description):
    """Run a single test query and display results"""
    print_header(f"TEST {query_num}: {description}")
    print(f"Query: {query_text}\n")
    
    try:
        response = orchestrator.process_query(query_text, verbose=True)
        formatted_output = orchestrator.synthesis_agent.format_for_display(response)
        print(formatted_output)
        return True
    except Exception as e:
        print(f"\n Error: {e}")
        return False


def main():
    # Setup log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(__file__).parent / f"test_results_{timestamp}.log"
    
    # Redirect output to both console and log file
    log_capture = LogCapture(log_file)
    sys.stdout = log_capture
    
    print_header("MedRAG System Testing - Example Queries")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log File: {log_file.name}")
    
    # Initialize system
    try:
        print("\nInitializing MedRAG system...")
        orchestrator = MedRAGOrchestrator()
    except Exception as e:
        print(f" Failed to initialize system: {e}")
        log_capture.close()
        sys.stdout = log_capture.terminal
        return
    
    # Define test queries based on actual data
    test_queries = [
        {
            "num": 1,
            "description": "Symptom-to-Disease Query (Knowledge Graph)",
            "query": "I am experiencing intense itching, especially at night, and small blisters on my skin. What could this be?",
            "expected": "Should identify Scabies from KG"
        },
        {
            "num": 2,
            "description": "Medicine Information Query (Vector Database)",
            "query": "Tell me about Huminsulin N 40IU/ml Injection",
            "expected": "Should retrieve Huminsulin insulin information from Vector DB"
        },
        {
            "num": 3,
            "description": "Medicine Side Effects Query (Vector Database)",
            "query": "What is Crocin 650mg Advance Tablet used for?",
            "expected": "Should retrieve Crocin medicine information from Vector DB"
        },
        {
            "num": 4,
            "description": "Multi-Symptom Query (Knowledge Graph Matching)",
            "query": "I have abdominal pain and bleeding. What conditions could cause these symptoms?",
            "expected": "Should find diseases matching multiple symptoms (e.g., Internal Organ Injury)"
        }
    ]
    
    # Run all tests
    results = []
    for test in test_queries:
        print(f"\nExpected Behavior: {test['expected']}")
        success = run_test_query(
            orchestrator,
            test['num'],
            test['query'],
            test['description']
        )
        results.append({
            "test": test['num'],
            "description": test['description'],
            "success": success
        })
        
        # Add separator between tests
        print("\n" + "-"*80 + "\n")
    
    # Print summary
    print_header("TEST SUMMARY")
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"\nSuccess Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nTest Details:")
    for result in results:
        status = " PASS" if result['success'] else " FAIL"
        print(f"  Test {result['test']}: {status} - {result['description']}")
    
    print(f"\n Full results saved to: {log_file.name}")
    print("\n" + "="*80)
    
    # Cleanup
    log_capture.close()
    sys.stdout = log_capture.terminal
    
    print(f"\n Testing complete! Results saved to: {log_file}")
    print(f"   Location: {log_file.absolute()}")


if __name__ == "__main__":
    main()
