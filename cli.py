from app.orchestrator import Orchestrator

orch = Orchestrator()
print("Clinical RAG CLI - type 'quit' to exit")

while True:
    q = input("\nQuery: ").strip()
    if q.lower() in ['quit', 'exit', 'q']:
        break
    if not q:
        continue
    r = orch.process_query(q)
    print("\nAnswer:", r['answer'])
    print("Grounded:", r['grounded'], "| Sources:", len(r['sources']))
    if r['sources']:
        print("Sources:")
        for s in r['sources']:
            print(f"  - {s['metadata']['source']} (Patient: {s['metadata']['patient_id']})")