import re
from pydantic import BaseModel
from typing import List, Optional

class Plan(BaseModel):
    intent: str
    rewritten_query: str
    subqueries: List[str]
    target_sources: List[str]
    patient_id_filter: Optional[str] = None

class PlannerAgent:
    def __init__(self):
        pass

    def plan_query(self, user_query: str) -> Plan:
        query_lower = user_query.lower()
        
        # Extract Patient ID using Regex (e.g., P0001)
        id_match = re.search(r'[pP]\d{4}', user_query)
        patient_id = id_match.group(0).upper() if id_match else None

        intent = "general_inquiry"
        target_sources = ["clinical_notes"]
        subqueries = [user_query]

        # Route to appropriate data sources based on query keywords
        if "risk" in query_lower or "readmission" in query_lower:
            intent = "risk_assessment"
            target_sources = ["encounters", "labs", "diagnoses", "clinical_notes"]

        elif "summarize" in query_lower or "history" in query_lower:
            intent = "patient_summarization"
            target_sources = ["medications", "diagnoses", "labs", "encounters", "clinical_notes"]

        elif "medication" in query_lower or "meds" in query_lower or "drug" in query_lower or "prescription" in query_lower:
            intent = "medication_inquiry"
            target_sources = ["medications", "clinical_notes"]

        elif "diagnos" in query_lower or "condition" in query_lower or "disease" in query_lower or "chronic" in query_lower:
            intent = "diagnosis_inquiry"
            target_sources = ["diagnoses", "clinical_notes"]

        elif "lab" in query_lower or "blood" in query_lower or "test result" in query_lower or "abnormal" in query_lower:
            intent = "lab_inquiry"
            target_sources = ["labs", "clinical_notes"]

        elif "encounter" in query_lower or "visit" in query_lower or "admission" in query_lower or "hospital" in query_lower:
            intent = "encounter_inquiry"
            target_sources = ["encounters", "clinical_notes"]

        target_sources = list(set(target_sources))

        return Plan(
            intent=intent,
            rewritten_query=user_query,
            subqueries=subqueries,
            target_sources=target_sources,
            patient_id_filter=patient_id
        )
