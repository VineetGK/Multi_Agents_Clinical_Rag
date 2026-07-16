import pandas as pd
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class ClinicalAnalyzer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.patients_df = pd.read_csv(os.path.join(data_dir, "patients.csv"))
        self.diagnoses_df = pd.read_csv(os.path.join(data_dir, "diagnoses.csv"))
        self.labs_df = pd.read_csv(os.path.join(data_dir, "labs.csv"))
        self.medications_df = pd.read_csv(os.path.join(data_dir, "medications.csv"))
        self.encounters_df = pd.read_csv(os.path.join(data_dir, "encounters.csv"))
        self.clinical_notes_df = pd.read_csv(os.path.join(data_dir, "clinical_notes.csv"))

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        p = self.patients_df[self.patients_df['patient_id'] == patient_id]
        if p.empty:
            return None
        return p.iloc[0].to_dict()

    def get_diagnoses(self, patient_id: str) -> pd.DataFrame:
        return self.diagnoses_df[self.diagnoses_df['patient_id'] == patient_id].sort_values('diagnosis_date')

    def get_labs(self, patient_id: str) -> pd.DataFrame:
        return self.labs_df[self.labs_df['patient_id'] == patient_id].sort_values('result_date')

    def get_medications(self, patient_id: str) -> pd.DataFrame:
        return self.medications_df[self.medications_df['patient_id'] == patient_id].sort_values('start_date')

    def get_encounters(self, patient_id: str) -> pd.DataFrame:
        return self.encounters_df[self.encounters_df['patient_id'] == patient_id].sort_values('admit_date')

    def get_clinical_notes(self, patient_id: str) -> pd.DataFrame:
        return self.clinical_notes_df[self.clinical_notes_df['patient_id'] == patient_id].sort_values('note_date')

    def answer_patient_questions(self, patient_id: str) -> Dict[str, Any]:
        patient = self.get_patient(patient_id)
        if not patient:
            return {"error": f"Patient {patient_id} not found"}

        diagnoses = self.get_diagnoses(patient_id)
        labs = self.get_labs(patient_id)
        medications = self.get_medications(patient_id)
        encounters = self.get_encounters(patient_id)
        notes = self.get_clinical_notes(patient_id)

        return {
            "patient_id": patient_id,
            "demographics": self._format_demographics(patient),
            "diagnoses": self._format_diagnoses(diagnoses),
            "labs": self._format_labs(labs),
            "medications": self._format_medications(medications),
            "encounters": self._format_encounters(encounters),
            "clinical_notes": self._format_notes(notes),
            "analysis": {
                "readmission_risk": self._analyze_readmission_risk(encounters, notes, diagnoses, labs),
                "diabetes_control": self._analyze_diabetes_control(labs, diagnoses),
                "renal_function": self._analyze_renal_function(labs, diagnoses),
                "heart_failure": self._analyze_heart_failure(labs, diagnoses, notes),
                "anemia": self._analyze_anemia(labs),
                "infection_inflammation": self._analyze_infection(labs),
                "multimorbidity": self._analyze_multimorbidity(diagnoses, notes),
                "timeline": self._build_timeline(diagnoses, labs, encounters, medications, notes),
                "followup_30day": self._analyze_30day_followup(encounters, labs, medications, notes),
            }
        }

    def _format_demographics(self, patient: Dict) -> str:
        return f"Age: {patient['age']}, Sex: {patient['sex']}, Race: {patient['race']}, Chronic Conditions Count: {patient['chronic_conditions_count']}"

    def _format_diagnoses(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No diagnoses recorded"
        lines = []
        for _, row in df.iterrows():
            lines.append(f"- {row['diagnosis_date']}: {row['diagnosis_name']} (ICD: {row['icd_code']})")
        return "\n".join(lines)

    def _format_labs(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No lab results recorded"
        lines = []
        for _, row in df.iterrows():
            lines.append(f"- {row['result_date']}: {row['lab_name']} = {row['lab_value']} {row['lab_unit']} ({row['abnormal_flag']})")
        return "\n".join(lines)

    def _format_medications(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No medications recorded"
        lines = []
        for _, row in df.iterrows():
            status = "Active" if row['active_flag'] == 1 else "Inactive"
            lines.append(f"- {row['start_date']}: {row['medication_name']} ({row['class']}) - Status: {status}")
        return "\n".join(lines)

    def _format_encounters(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No encounters recorded"
        lines = []
        for _, row in df.iterrows():
            risk = "HIGH RISK" if row.get('readmission_30d', 0) == 1 else "Standard risk"
            lines.append(f"- {row['admit_date']} to {row['discharge_date']}: {row['encounter_type']} ({risk})")
        return "\n".join(lines)

    def _format_notes(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No clinical notes recorded"
        lines = []
        for _, row in df.iterrows():
            lines.append(f"- {row['note_date']} [{row['note_type']}]: {row['note_text']}")
        return "\n".join(lines)

    def _analyze_readmission_risk(self, encounters, notes, diagnoses, labs) -> str:
        high_risk_encounters = encounters[encounters['readmission_30d'] == 1]
        risk_notes = notes[notes['note_text'].str.contains('high risk|readmission', case=False, na=False)]
        chronic_count = diagnoses['diagnosis_name'].nunique()
        abnormal_labs = labs[labs['abnormal_flag'] == 'ABNORMAL']
        
        parts = []
        if not high_risk_encounters.empty:
            parts.append(f"High-risk encounters: {len(high_risk_encounters)} (most recent: {high_risk_encounters.iloc[-1]['admit_date']})")
        if not risk_notes.empty:
            parts.append(f"Clinical notes flagging readmission risk: {len(risk_notes)}")
        parts.append(f"Chronic conditions: {chronic_count}")
        parts.append(f"Abnormal labs: {len(abnormal_labs)}")
        return "; ".join(parts)

    def _analyze_diabetes_control(self, labs, diagnoses) -> str:
        has_diabetes = diagnoses['diagnosis_name'].str.contains('diabetes', case=False).any()
        a1c_labs = labs[labs['lab_name'].str.contains('A1c|HbA1c', case=False, na=False)]
        glucose_labs = labs[labs['lab_name'].str.contains('glucose', case=False, na=False)]
        
        if not has_diabetes and a1c_labs.empty and glucose_labs.empty:
            return "No diabetes diagnosis or related labs found"
        
        parts = []
        if has_diabetes:
            parts.append("Diabetes diagnosis: YES")
        if not a1c_labs.empty:
            latest_a1c = a1c_labs.sort_values('result_date').iloc[-1]
            parts.append(f"Latest A1c: {latest_a1c['lab_value']}% on {latest_a1c['result_date']} ({latest_a1c['abnormal_flag']})")
            if len(a1c_labs) > 1:
                first = a1c_labs.sort_values('result_date').iloc[0]
                last = a1c_labs.sort_values('result_date').iloc[-1]
                trend = "improving" if float(last['lab_value']) < float(first['lab_value']) else "worsening"
                parts.append(f"A1c trend: {first['lab_value']}% -> {last['lab_value']}% ({trend})")
        if not glucose_labs.empty:
            latest_glu = glucose_labs.sort_values('result_date').iloc[-1]
            parts.append(f"Latest Glucose: {latest_glu['lab_value']} {latest_glu['lab_unit']} on {latest_glu['result_date']} ({latest_glu['abnormal_flag']})")
        return "; ".join(parts)

    def _analyze_renal_function(self, labs, diagnoses) -> str:
        has_ckd = diagnoses['diagnosis_name'].str.contains('ckd|chronic kidney', case=False).any()
        creat_labs = labs[labs['lab_name'].str.contains('creatinine', case=False, na=False)]
        
        if not has_ckd and creat_labs.empty:
            return "No CKD diagnosis or creatinine labs found"
        
        parts = []
        if has_ckd:
            parts.append("CKD diagnosis: YES")
        if not creat_labs.empty:
            latest = creat_labs.sort_values('result_date').iloc[-1]
            parts.append(f"Latest Creatinine: {latest['lab_value']} {latest['lab_unit']} on {latest['result_date']} ({latest['abnormal_flag']})")
            abnormal_count = len(creat_labs[creat_labs['abnormal_flag'] == 'ABNORMAL'])
            if abnormal_count > 1:
                parts.append(f"Multiple abnormal creatinine results: {abnormal_count}")
        return "; ".join(parts)

    def _analyze_heart_failure(self, labs, diagnoses, notes) -> str:
        has_chf = diagnoses['diagnosis_name'].str.contains('chf|heart failure', case=False).any()
        bnp_labs = labs[labs['lab_name'].str.contains('bnp', case=False, na=False)]
        hf_notes = notes[notes['note_text'].str.contains('heart failure|chf|bnp', case=False, na=False)]
        
        if not has_chf and bnp_labs.empty and hf_notes.empty:
            return "No heart failure evidence found"
        
        parts = []
        if has_chf:
            parts.append("CHF/Heart Failure diagnosis: YES")
        if not bnp_labs.empty:
            latest = bnp_labs.sort_values('result_date').iloc[-1]
            parts.append(f"Latest BNP: {latest['lab_value']} {latest['lab_unit']} on {latest['result_date']} ({latest['abnormal_flag']})")
        if not hf_notes.empty:
            parts.append(f"Clinical notes mentioning HF: {len(hf_notes)}")
        return "; ".join(parts)

    def _analyze_anemia(self, labs) -> str:
        hgb_labs = labs[labs['lab_name'].str.contains('hemoglobin|hgb', case=False, na=False)]
        if hgb_labs.empty:
            return "No hemoglobin labs found"
        latest = hgb_labs.sort_values('result_date').iloc[-1]
        abnormal = hgb_labs[hgb_labs['abnormal_flag'] == 'ABNORMAL']
        parts = [f"Latest Hemoglobin: {latest['lab_value']} {latest['lab_unit']} on {latest['result_date']} ({latest['abnormal_flag']})"]
        if len(abnormal) > 0:
            parts.append(f"Abnormal hemoglobin results: {len(abnormal)}")
        return "; ".join(parts)

    def _analyze_infection(self, labs) -> str:
        wbc_labs = labs[labs['lab_name'].str.contains('wbc|white blood', case=False, na=False)]
        if wbc_labs.empty:
            return "No WBC labs found"
        latest = wbc_labs.sort_values('result_date').iloc[-1]
        abnormal = wbc_labs[wbc_labs['abnormal_flag'] == 'ABNORMAL']
        parts = [f"Latest WBC: {latest['lab_value']} {latest['lab_unit']} on {latest['result_date']} ({latest['abnormal_flag']})"]
        if len(abnormal) > 0:
            parts.append(f"Abnormal WBC results: {len(abnormal)} (suggesting possible infection/inflammation)")
        return "; ".join(parts)

    def _analyze_multimorbidity(self, diagnoses, notes) -> str:
        unique_dx = diagnoses['diagnosis_name'].nunique()
        chronic_keywords = ['diabetes', 'copd', 'ckd', 'chf', 'hypertension', 'heart failure', 'kidney', 'depression']
        chronic_count = sum(1 for d in diagnoses['diagnosis_name'].unique() if any(k in d.lower() for k in chronic_keywords))
        return f"Unique diagnoses: {unique_dx}; Chronic conditions identified: {chronic_count} (multimorbidity: {'YES' if chronic_count >= 2 else 'NO'})"

    def _build_timeline(self, diagnoses, labs, encounters, medications, notes) -> List[Dict]:
        events = []
        for _, r in diagnoses.iterrows():
            events.append({"date": r['diagnosis_date'], "type": "Diagnosis", "detail": f"{r['diagnosis_name']} ({r['icd_code']})"})
        for _, r in labs.iterrows():
            events.append({"date": r['result_date'], "type": "Lab", "detail": f"{r['lab_name']}={r['lab_value']}{r['lab_unit']} ({r['abnormal_flag']})"})
        for _, r in encounters.iterrows():
            events.append({"date": r['admit_date'], "type": "Encounter", "detail": f"{r['encounter_type']} ({r['admit_date']} to {r['discharge_date']})"})
        for _, r in medications.iterrows():
            status = "Active" if r['active_flag'] == 1 else "Inactive"
            events.append({"date": r['start_date'], "type": "Medication", "detail": f"{r['medication_name']} ({r['class']}) - {status}"})
        for _, r in notes.iterrows():
            events.append({"date": r['note_date'], "type": "Note", "detail": f"[{r['note_type']}] {r['note_text'][:100]}"})
        events.sort(key=lambda x: x['date'])
        return events

    def _analyze_30day_followup(self, encounters, labs, medications, notes) -> str:
        if encounters.empty:
            return "No encounters to analyze"
        last_discharge = encounters.iloc[-1]['discharge_date']
        followup_enc = encounters[encounters['admit_date'] > last_discharge]
        followup_labs = labs[labs['result_date'] > last_discharge]
        followup_meds = medications[medications['start_date'] > last_discharge]
        followup_notes = notes[notes['note_date'] > last_discharge]
        
        parts = [f"Last discharge: {last_discharge}"]
        if not followup_enc.empty:
            parts.append(f"Follow-up encounters within 30 days: {len(followup_enc)}")
        if not followup_labs.empty:
            parts.append(f"Follow-up labs within 30 days: {len(followup_labs)}")
        if not followup_meds.empty:
            parts.append(f"New medications within 30 days: {len(followup_meds)}")
        if not followup_notes.empty:
            parts.append(f"Follow-up notes within 30 days: {len(followup_notes)}")
        if len(parts) == 1:
            parts.append("No follow-up activity within 30 days of last discharge")
        return "; ".join(parts)


def generate_handoff_summary(analysis: Dict) -> str:
    p = analysis
    demo = p['demographics']
    dx = p['analysis']
    
    lines = [
        f"HANDOFF SUMMARY - Patient {p['patient_id']}",
        f"Demographics: {demo}",
        f"Major Chronic Conditions: {dx['multimorbidity']}",
        f"Diabetes Control: {dx['diabetes_control']}",
        f"Renal Function: {dx['renal_function']}",
        f"Heart Failure Evidence: {dx['heart_failure']}",
        f"Anemia: {dx['anemia']}",
        f"Infection/Inflammation: {dx['infection_inflammation']}",
        f"Readmission Risk Signals: {dx['readmission_risk']}",
        f"Recent Timeline: {len(dx['timeline'])} events from {dx['timeline'][0]['date'] if dx['timeline'] else 'N/A'} to {dx['timeline'][-1]['date'] if dx['timeline'] else 'N/A'}",
        f"30-day Follow-up: {dx['followup_30day']}",
    ]
    return "\n".join(lines)