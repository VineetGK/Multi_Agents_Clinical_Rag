from typing import List, Dict, Any, Tuple

class EvaluatorAgent:
    def __init__(self):
        pass

    def evaluate_answer(self, user_query: str, draft_answer: str, contexts: List[Dict[str, Any]]) -> Tuple[bool, List[str], bool]:
        # In a real system, this would prompt an LLM to check for groundedness.
        # Here we simulate the evaluation.
        issues = []
        suggested_retry = False

        if not draft_answer or len(draft_answer.strip()) < 10:
            issues.append("Answer is too short or empty.")
            return False, issues, True

        if not contexts:
            issues.append("No context was provided to ground the answer.")
            return False, issues, True

        # Check if patient IDs or specific keywords from the context are in the answer
        context_text = " ".join([c['text'].lower() for c in contexts])

        # Simple heuristic: if the draft answer mentions "high risk" but the contexts don't,
        # flag it as ungrounded.
        if "high risk" in draft_answer.lower() and "high risk" not in context_text:
            issues.append("The claim of 'high risk' is not supported by the retrieved context.")
            suggested_retry = True

        if issues:
            return False, issues, suggested_retry

        return True, ["Answer is well-grounded in the provided context."], False
