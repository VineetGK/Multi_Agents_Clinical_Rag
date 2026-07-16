import pandas as pd
import os
import re
from .planner_agent import PlannerAgent
from .retrieval_agent import RetrievalAgent
from .evaluator_agent import EvaluatorAgent
from .clinical_analyzer import ClinicalAnalyzer, generate_handoff_summary

class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrievalAgent()
        self.evaluator = EvaluatorAgent()
        self.analyzer = ClinicalAnalyzer()
        self.patients_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'patients.csv'))

    def _is_complex_clinical_query(self, query: str) -> bool:
        query_lower = query.lower()
        clinical_keywords = [
            'handoff', 'summarize', 'timeline', 'trend', 'readmission risk', 'readmission factors',
            'diabetes control', 'renal dysfunction', 'anemia', 'heart failure',
            'chronic kidney disease', 'ckd', 'multimorbidity', 'follow-up', 'follow up', '30 day', '30-day',
            'abnormal creatinine', 'abnormal bnp', 'abnormal a1c', 'abnormal glucose',
            'abnormal sodium', 'abnormal wbc', 'abnormal hemoglobin',
            'creatinine trend', 'bnp trend', 'a1c trend', 'glucose trend', 'hba1c',
            'repeated elevated', 'multiple elevated', 'infection', 'inflammation',
            'discharge summary', 'clinical note', 'encounter', 'medication',
            'what diagnoses', 'what labs', 'what medications', 'what happened',
            'most recent', 'latest', 'ever been labeled', 'high risk for readmission',
            'age', 'sex', 'race', 'chronic conditions', 'demographics',
            'poorly controlled', 'worsening', 'improving', 'align', 'handoff summary',
            'labeled high risk', 'high risk', 'readmission'
        ]
        has_patient = bool(re.search(r'[pP]\d{4}', query))
        return has_patient and any(kw in query_lower for kw in clinical_keywords)

    def _is_count_query(self, query: str) -> bool:
        query_lower = query.lower()
        count_keywords = ['how many', 'count', 'number of', 'total']
        is_count = any(kw in query_lower for kw in count_keywords)
        has_patient_id = bool(re.search(r'[pP]\d{4}', query))
        return is_count and not has_patient_id

    def _handle_complex_clinical_query(self, query: str) -> dict:
        patient_match = re.search(r'[pP]\d{4}', query)
        if not patient_match:
            return None
        patient_id = patient_match.group(0).upper()
        
        # Check if patient exists
        patient_row = self.patients_df[self.patients_df['patient_id'] == patient_id]
        if patient_row.empty:
            return {"answer": f"Patient {patient_id} not found in database.", "grounded": False, "sources": []}
        
        # Run clinical analysis
        analysis = self.analyzer.answer_patient_questions(patient_id)
        
        # Generate response based on query type
        query_lower = query.lower()
        
        # 1. Demographics: age, sex, race, chronic conditions count
        if any(kw in query_lower for kw in ['age', 'sex', 'race', 'demographic', 'chronic condition count', 'chronic conditions count']):
            patient_row = self.patients_df[self.patients_df['patient_id'] == patient_id].iloc[0]
            return {
                "answer": f"Patient {patient_id}: Age {patient_row['age']}, Sex {patient_row['sex']}, Race {patient_row['race']}, Chronic Conditions Count: {patient_row['chronic_conditions_count']}",
                "grounded": True,
                "sources": [{"text": f"From patients.csv", "metadata": {"source": "patients.csv", "patient_id": patient_id}}]
            }
        
        # 2. Diagnoses with dates
        if 'diagnos' in query_lower:
            dx_str = analysis['diagnoses']
            if dx_str and dx_str != "No diagnoses recorded":
                ans = f"Patient {patient_id} diagnoses:\n{dx_str}"
            else:
                ans = f"No diagnoses found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From diagnoses.csv", "metadata": {"source": "diagnoses.csv", "patient_id": patient_id}}]}
        
        # 3. High readmission risk labels in clinical notes (must come before discharge summary)
        if any(kw in query_lower for kw in ['readmission', 'high risk']) and any(kw in query_lower for kw in ['label', 'ever been labeled', 'flagged', 'explicit', 'labeled']):
            notes_str = analysis['clinical_notes']
            if notes_str and notes_str != "No clinical notes recorded":
                risk_lines = [n for n in notes_str.split('\n') if 'high risk' in n.lower() or 'readmission' in n.lower()]
                if risk_lines:
                    ans = f"Patient {patient_id} has been labeled high risk for readmission:\n" + "\n".join(risk_lines)
                else:
                    ans = f"Patient {patient_id} has no explicit high readmission risk labels in clinical notes."
            else:
                ans = f"No clinical notes found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From clinical_notes.csv", "metadata": {"source": "clinical_notes.csv", "patient_id": patient_id}}]}
        
        # 4. Discharge summaries / clinical notes
        if any(kw in query_lower for kw in ['discharge', 'clinical note', 'summary']):
            notes_str = analysis['clinical_notes']
            if notes_str and notes_str != "No clinical notes recorded":
                ans = f"Patient {patient_id} discharge summaries:\n{notes_str}"
            else:
                ans = f"No discharge summaries found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From clinical_notes.csv", "metadata": {"source": "clinical_notes.csv", "patient_id": patient_id}}]}
        
        # 5. Latest lab results with abnormal flags
        if 'lab' in query_lower and any(kw in query_lower for kw in ['latest', 'recent', 'abnormal']):
            labs_str = analysis['labs']
            if labs_str and labs_str != "No lab results recorded":
                ans = f"Patient {patient_id} lab results:\n{labs_str}"
            else:
                ans = f"No lab results found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From labs.csv", "metadata": {"source": "labs.csv", "patient_id": patient_id}}]}
        
        # 6. Specific lab tests (creatinine, BNP, A1c, glucose, sodium, WBC, hemoglobin)
        if any(x in query_lower for x in ['creatinine', 'bnp', 'a1c', 'hba1c', 'glucose', 'sodium', 'wbc', 'white blood', 'hemoglobin', 'hgb']):
            labs_str = analysis['labs']
            if labs_str and labs_str != "No lab results recorded":
                lab_lines = labs_str.split('\n')
                test_keywords = []
                if 'creatinine' in query_lower: test_keywords.append('creatinine')
                if 'bnp' in query_lower: test_keywords.append('bnp')
                if 'a1c' in query_lower or 'hba1c' in query_lower: test_keywords.append('a1c')
                if 'a1c' in query_lower or 'hba1c' in query_lower: test_keywords.append('hba1c')
                if 'glucose' in query_lower: test_keywords.append('glucose')
                if 'sodium' in query_lower: test_keywords.append('sodium')
                if 'wbc' in query_lower or 'white blood' in query_lower: test_keywords.append('wbc')
                if 'hemoglobin' in query_lower or 'hgb' in query_lower: test_keywords.append('hemoglobin')
                
                matched = [l for l in lab_lines if any(kw in l.lower() for kw in test_keywords)]
                if matched:
                    ans = f"Patient {patient_id} {', '.join(test_keywords).upper()} results:\n" + "\n".join(matched)
                else:
                    ans = f"No {', '.join(test_keywords).upper()} results found for patient {patient_id}."
            else:
                ans = f"No lab results found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From labs.csv", "metadata": {"source": "labs.csv", "patient_id": patient_id}}]}
        
        # 7. A1c trend with diabetes control assessment
        if 'a1c' in query_lower and 'trend' in query_lower:
            labs_str = analysis['labs']
            if labs_str and labs_str != "No lab results recorded":
                lab_lines = labs_str.split('\n')
                a1c_lines = [l for l in lab_lines if 'a1c' in l.lower() or 'hba1c' in l.lower()]
                if a1c_lines:
                    ans = f"Patient {patient_id} A1c trend:\n" + "\n".join(a1c_lines)
                    # Parse values for trend
                    values = []
                    for l in a1c_lines:
                        try:
                            val = float(l.split('=')[1].split('%')[0].strip())
                            values.append(val)
                        except:
                            pass
                    if len(values) >= 2:
                        first = values[0]
                        last = values[-1]
                        if last > first:
                            ans += f"\nDiabetes control appears to be WORSENING (A1c increased from {first}% to {last}%)."
                        elif last < first:
                            ans += f"\nDiabetes control appears to be IMPROVING (A1c decreased from {first}% to {last}%)."
                        else:
                            ans += f"\nDiabetes control appears STABLE (A1c unchanged at {first}%)."
                else:
                    ans = f"No A1c results found for patient {patient_id}."
            else:
                ans = f"No lab results found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From labs.csv", "metadata": {"source": "labs.csv", "patient_id": patient_id}}]}
        
        # 8. Repeated renal dysfunction / multiple elevated creatinine
        if 'creatinine' in query_lower and any(kw in query_lower for kw in ['repeated', 'multiple', 'renal dysfunction', 'elevated']):
            labs_str = analysis['labs']
            if labs_str and labs_str != "No lab results recorded":
                lab_lines = labs_str.split('\n')
                cr_lines = [l for l in lab_lines if 'creatinine' in l.lower()]
                elevated = [l for l in cr_lines if '(abnormal)' in l.lower() or '(true)' in l.lower()]
                if elevated:
                    ans = f"Patient {patient_id} has {len(elevated)} elevated creatinine results (suggesting repeated renal dysfunction):\n" + "\n".join(elevated)
                else:
                    ans = f"No elevated creatinine results found for patient {patient_id}."
            else:
                ans = f"No lab results found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From labs.csv", "metadata": {"source": "labs.csv", "patient_id": patient_id}}]}
        
        # 9. Infection/inflammation from abnormal WBC
        if 'wbc' in query_lower and any(kw in query_lower for kw in ['infection', 'inflammation', 'abnormal']):
            labs_str = analysis['labs']
            if labs_str and labs_str != "No lab results recorded":
                lab_lines = labs_str.split('\n')
                wbc_lines = [l for l in lab_lines if 'wbc' in l.lower() or 'white blood' in l.lower()]
                abnormal = [l for l in wbc_lines if '(abnormal)' in l.lower() or '(true)' in l.lower()]
                if abnormal:
                    ans = f"Patient {patient_id} has abnormal WBC results (possible infection/inflammation):\n" + "\n".join(abnormal)
                else:
                    ans = f"No abnormal WBC results found for patient {patient_id}."
            else:
                ans = f"No lab results found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From labs.csv", "metadata": {"source": "labs.csv", "patient_id": patient_id}}]}
        
        # 10. Anemia from hemoglobin history
        if 'hemoglobin' in query_lower and any(kw in query_lower for kw in ['anemia', 'trend', 'history']):
            labs_str = analysis['labs']
            if labs_str and labs_str != "No lab results recorded":
                lab_lines = labs_str.split('\n')
                hgb_lines = [l for l in lab_lines if 'hemoglobin' in l.lower() or 'hgb' in l.lower()]
                if hgb_lines:
                    ans = f"Patient {patient_id} Hemoglobin trend:\n" + "\n".join(hgb_lines)
                    abnormal_count = sum(1 for l in hgb_lines if '(abnormal)' in l.lower() or '(true)' in l.lower())
                    if abnormal_count > 0:
                        ans += f"\nPossible anemia indicated ({abnormal_count} abnormal result(s))."
                else:
                    ans = f"No hemoglobin results found for patient {patient_id}."
            else:
                ans = f"No lab results found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From labs.csv", "metadata": {"source": "labs.csv", "patient_id": patient_id}}]}
        
        # 11. Heart failure evidence (CHF diagnosis, elevated BNP, discharge notes)
        if any(kw in query_lower for kw in ['heart failure', 'chf', 'bnp']):
            dx_str = analysis['diagnoses']
            labs_str = analysis['labs']
            notes_str = analysis['clinical_notes']
            
            chf_dx = [l for l in dx_str.split('\n') if 'chf' in l.lower() or 'heart failure' in l.lower()] if dx_str != "No diagnoses recorded" else []
            bnp_lines = [l for l in labs_str.split('\n') if 'bnp' in l.lower()] if labs_str != "No lab results recorded" else []
            hf_notes = [n for n in notes_str.split('\n') if 'heart failure' in n.lower() or 'chf' in n.lower()] if notes_str != "No clinical notes recorded" else []
            
            ans = f"Patient {patient_id} Heart Failure Evidence:\n"
            if chf_dx:
                ans += f"CHF Diagnoses: {len(chf_dx)}\n" + "\n".join(chf_dx)
            else:
                ans += "No CHF diagnosis found\n"
            if bnp_lines:
                ans += f"\nBNP Results: {len(bnp_lines)} test(s)\n" + "\n".join(bnp_lines)
            else:
                ans += "\nNo BNP results found"
            if hf_notes:
                ans += f"\nClinical notes mentioning HF: {len(hf_notes)}\n" + "\n".join(hf_notes)
            return {"answer": ans, "grounded": True, "sources": [{"text": "From diagnoses.csv, labs.csv, clinical_notes.csv", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 12. Chronic kidney disease (CKD diagnosis + creatinine)
        if any(kw in query_lower for kw in ['kidney disease', 'ckd', 'chronic kidney']) or ('renal' in query_lower and 'creatinine' in query_lower):
            dx_str = analysis['diagnoses']
            labs_str = analysis['labs']
            
            ckd_dx = [l for l in dx_str.split('\n') if 'ckd' in l.lower() or 'chronic kidney' in l.lower() or 'renal' in l.lower()] if dx_str != "No diagnoses recorded" else []
            cr_lines = [l for l in labs_str.split('\n') if 'creatinine' in l.lower()] if labs_str != "No lab results recorded" else []
            
            ans = f"Patient {patient_id} CKD Evidence:\n"
            if ckd_dx:
                ans += f"CKD Diagnosis: {len(ckd_dx)} found\n" + "\n".join(ckd_dx)
            else:
                ans += "No CKD diagnosis found\n"
            if cr_lines:
                ans += f"\nCreatinine Results: {len(cr_lines)} test(s)\n" + "\n".join(cr_lines)
            else:
                ans += "\nNo creatinine results found"
            return {"answer": ans, "grounded": True, "sources": [{"text": "From diagnoses.csv, labs.csv", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 13. Poorly controlled diabetes (diagnoses + glucose + A1c)
        if 'diabetes' in query_lower and any(kw in query_lower for kw in ['control', 'poorly', 'glucose', 'a1c', 'hba1c']):
            dx_str = analysis['diagnoses']
            labs_str = analysis['labs']
            
            dm_dx = [l for l in dx_str.split('\n') if 'diabetes' in l.lower()] if dx_str != "No diagnoses recorded" else []
            a1c_lines = [l for l in labs_str.split('\n') if 'a1c' in l.lower() or 'hba1c' in l.lower()] if labs_str != "No lab results recorded" else []
            glu_lines = [l for l in labs_str.split('\n') if 'glucose' in l.lower()] if labs_str != "No lab results recorded" else []
            
            ans = f"Patient {patient_id} Diabetes Control:\n"
            if dm_dx:
                ans += f"Diabetes Diagnosis: {len(dm_dx)} found\n" + "\n".join(dm_dx)
            else:
                ans += "No diabetes diagnosis found\n"
            if a1c_lines:
                ans += f"\nA1c Results:\n" + "\n".join(a1c_lines)
            if glu_lines:
                ans += f"\nGlucose Results (latest 5):\n" + "\n".join(glu_lines[-5:])
            return {"answer": ans, "grounded": True, "sources": [{"text": "From diagnoses.csv, labs.csv", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 14. Multimorbidity
        if 'multimorbidity' in query_lower or 'multiple chronic' in query_lower:
            dx_str = analysis['diagnoses']
            if dx_str != "No diagnoses recorded":
                dx_lines = dx_str.split('\n')
                unique_conditions = set()
                for l in dx_lines:
                    # Extract condition name (between date and ICD)
                    if ':' in l and '(' in l:
                        cond = l.split(':')[1].split('(')[0].strip().lower()
                        unique_conditions.add(cond)
                ans = f"Patient {patient_id} Multimorbidity Assessment:\n"
                ans += f"Unique diagnosed conditions: {len(unique_conditions)}\n"
                for c in sorted(unique_conditions):
                    ans += f"  - {c}\n"
                if len(unique_conditions) >= 2:
                    ans += "MULTIMORBIDITY CONFIRMED (2+ chronic conditions)"
                else:
                    ans += "Multimorbidity not confirmed (<2 conditions)"
            else:
                ans = f"No diagnoses found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From diagnoses.csv", "metadata": {"source": "diagnoses.csv", "patient_id": patient_id}}]}
        
        # 15. Recent clinical timeline
        if any(kw in query_lower for kw in ['timeline', 'what happened', 'recent clinical']):
            timeline = analysis['analysis']['timeline']
            if timeline:
                ans = f"Patient {patient_id} Recent Clinical Timeline:\n"
                for t in timeline[-20:]:
                    ans += f"- {t['date']} [{t['type']}]: {t['detail']}\n"
            else:
                ans = f"No timeline events found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From multiple sources", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 16. Encounter count and most recent
        if 'encounter' in query_lower and any(kw in query_lower for kw in ['how many', 'most recent', 'count', 'number']):
            encounters_str = analysis['encounters']
            if encounters_str and encounters_str != "No encounters recorded":
                enc_lines = encounters_str.split('\n')
                ans = f"Patient {patient_id} Encounters: {len(enc_lines)} total\n"
                most_recent = enc_lines[-1] if enc_lines else "Unknown"
                ans += f"Most recent: {most_recent}"
            else:
                ans = f"No encounters found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From encounters.csv", "metadata": {"source": "encounters.csv", "patient_id": patient_id}}]}
        
        # 17. Medications aligning with chronic conditions
        if 'medication' in query_lower and any(kw in query_lower for kw in ['align', 'chronic']):
            meds_str = analysis['medications']
            dx_str = analysis['diagnoses']
            if dx_str != "No diagnoses recorded":
                dx_lines = dx_str.split('\n')
                conditions = set()
                for l in dx_lines:
                    if ':' in l and '(' in l:
                        cond = l.split(':')[1].split('(')[0].strip().lower()
                        conditions.add(cond)
            else:
                conditions = set()
            
            ans = f"Patient {patient_id} Medications vs Conditions:\n"
            ans += f"Diagnosed conditions: {', '.join(sorted(conditions)) if conditions else 'None'}\n\n"
            ans += "Prescribed medications:\n"
            if meds_str and meds_str != "No medications recorded":
                ans += meds_str
            else:
                ans += "No medications recorded"
            return {"answer": ans, "grounded": True, "sources": [{"text": "From medications.csv, diagnoses.csv", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 18. 30-day follow-up after discharge
        if any(kw in query_lower for kw in ['follow-up', 'follow up', '30 day', '30-day']):
            followup = analysis['analysis']['followup_30day']
            if followup:
                ans = f"Patient {patient_id} 30-day follow-up after discharge:\n{followup}"
            else:
                ans = f"No follow-up events within 30 days of discharge found for patient {patient_id}."
            return {"answer": ans, "grounded": True, "sources": [{"text": "From encounters.csv, labs.csv, medications.csv", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 19. Readmission risk factors
        if 'readmission' in query_lower and 'factor' in query_lower:
            risk = analysis['analysis']['readmission_risk']
            ans = f"Patient {patient_id} Readmission Risk Factors:\n{risk}"
            return {"answer": ans, "grounded": True, "sources": [{"text": "From clinical_notes.csv, diagnoses.csv, labs.csv, encounters.csv", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        # 20. Handoff summary
        if 'handoff' in query_lower or 'summarize' in query_lower:
            handoff = generate_handoff_summary({'patient_id': patient_id, 'demographics': analysis['demographics'], 'analysis': analysis['analysis']})
            return {"answer": handoff, "grounded": True, "sources": [{"text": "From all data sources", "metadata": {"source": "multiple", "patient_id": patient_id}}]}
        
        return None

    def _handle_count_query(self, query: str) -> dict:
        if not self._is_count_query(query):
            return None
            
        query_lower = query.lower()
        
        # Age-based count queries
        import re
        age_below_match = re.search(r'(below|under|less than)\s+age?\s*(\d+)', query_lower)
        age_above_match = re.search(r'(above|over|greater than|older than)\s+age?\s*(\d+)', query_lower)
        age_between_match = re.search(r'(between|age)\s+(\d+)\s*(?:and|to|-)\s*(\d+)', query_lower)
        
        if age_below_match:
            age = int(age_below_match.group(2))
            count = len(self.patients_df[self.patients_df['age'] < age])
            return {"answer": f"There are {count} patients below age {age}.", "grounded": True, "sources": [{"text": f"Counted from patients.csv: {count} patients with age < {age}", "metadata": {"source": "patients.csv", "patient_id": "N/A"}}]}
        
        if age_above_match:
            age = int(age_above_match.group(2))
            count = len(self.patients_df[self.patients_df['age'] > age])
            return {"answer": f"There are {count} patients above age {age}.", "grounded": True, "sources": [{"text": f"Counted from patients.csv: {count} patients with age > {age}", "metadata": {"source": "patients.csv", "patient_id": "N/A"}}]}
        
        if age_between_match:
            age1 = int(age_between_match.group(2))
            age2 = int(age_between_match.group(3))
            count = len(self.patients_df[(self.patients_df['age'] >= age1) & (self.patients_df['age'] <= age2)])
            return {"answer": f"There are {count} patients between age {age1} and {age2}.", "grounded": True, "sources": [{"text": f"Counted from patients.csv: {count} patients with age between {age1} and {age2}", "metadata": {"source": "patients.csv", "patient_id": "N/A"}}]}
        
        if 'female' in query_lower and ('patient' in query_lower or 'patients' in query_lower):
            count = len(self.patients_df[self.patients_df['sex'] == 'F'])
            return {"answer": f"There are {count} female patients in the database.", "grounded": True, "sources": [{"text": f"Counted from patients.csv: {count} female patients", "metadata": {"source": "patients.csv", "patient_id": "N/A"}}]}
        
        if 'male' in query_lower and ('patient' in query_lower or 'patients' in query_lower):
            count = len(self.patients_df[self.patients_df['sex'] == 'M'])
            return {"answer": f"There are {count} male patients in the database.", "grounded": True, "sources": [{"text": f"Counted from patients.csv: {count} male patients", "metadata": {"source": "patients.csv", "patient_id": "N/A"}}]}
        
        if 'patient' in query_lower or 'patients' in query_lower:
            count = len(self.patients_df)
            return {"answer": f"There are {count} total patients in the database.", "grounded": True, "sources": [{"text": f"Counted from patients.csv: {count} total patients", "metadata": {"source": "patients.csv", "patient_id": "N/A"}}]}
        
        return None

    def process_query(self, query: str) -> dict:
        # Step 0: Handle count queries
        count_result = self._handle_count_query(query)
        if count_result:
            return {
                "query": query,
                "answer": count_result["answer"],
                "grounded": count_result["grounded"],
                "sources": count_result["sources"],
                "evaluation": {"issues": [], "suggested_retry": False}
            }

        # Step 0b: Handle complex clinical queries
        if self._is_complex_clinical_query(query):
            clinical_result = self._handle_complex_clinical_query(query)
            if clinical_result:
                return {
                    "query": query,
                    "answer": clinical_result["answer"],
                    "grounded": clinical_result["grounded"],
                    "sources": clinical_result["sources"],
                    "evaluation": {"issues": [], "suggested_retry": False}
                }

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
