import sys
from app.orchestrator import Orchestrator

orch = Orchestrator()

if len(sys.argv) < 2:
    print("Usage: python query.py \"your question here\"")
    print("Example: python query.py \"What medications is patient P0001 taking?\"")
    sys.exit(1)

query = " ".join(sys.argv[1:])
result = orch.process_query(query)

print(f"\nQuery: {query}")
print(f"Answer: {result['answer']}")
print(f"Grounded: {result['grounded']} | Sources: {len(result['sources'])}")
if result['sources']:
    print("Sources:")
    for s in result['sources']:
        print(f"  - {s['metadata']['source']} (Patient: {s['metadata']['patient_id']})")