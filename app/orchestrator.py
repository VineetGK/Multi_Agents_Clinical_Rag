from .planner_agent import PlannerAgent
from .retrieval_agent import RetrievalAgent
from .evaluator_agent import EvaluatorAgent

class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrievalAgent()
        self.evaluator = EvaluatorAgent()

    def process_query(self, query: str) -> dict:
        # Step 1: Plan
        plan = self.planner.plan_query(query)

        # Step 2: Retrieve
        contexts = self.retriever.retrieve_context(plan, top_k=5)

        # Step 3: Draft Answer
        draft_answer = self._generate_draft(query, contexts)

        # Step 4: Evaluate
        grounded, issues, retry = self.evaluator.evaluate_answer(query, draft_answer, contexts)

        # Step 5: Retry if needed
        if retry:
            contexts = self.retriever.retrieve_context(plan, top_k=10) # expand search
            draft_answer = self._generate_draft(query, contexts)
            grounded, issues, retry = self.evaluator.evaluate_answer(query, draft_answer, contexts)

        return {
            "query": query,
            "answer": draft_answer,
            "grounded": grounded,
            "sources": contexts,
            "evaluation": {
                "issues": issues,
                "suggested_retry": retry
            }
        }

    def _generate_draft(self, query: str, contexts: list) -> str:
        if not contexts:
            return "I could not find any relevant patient information to answer your query."

        # Simulated LLM generation based on context
        context_texts = "\n".join([f"- {c['text']} (Source: {c['metadata']['source']}, ID: {c['metadata']['patient_id']})" for c in contexts])
        return f"Based on the retrieved clinical records:\n{context_texts}\n\nThis evidence directly relates to: '{query}'."
