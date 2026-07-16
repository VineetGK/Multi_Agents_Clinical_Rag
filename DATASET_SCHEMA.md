# Clinical RAG System - Dataset Schema & Query Capabilities

## System Overview
A RAG-based clinical question-answering system processing structured EHR data for 500 patients. Uses a multi-agent orchestrator (Planner → Retriever → Evaluator) with a clinical analyzer for complex analytical queries.

---

## Dataset Schemas

### 1. patients.csv (500 rows)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | string | Unique identifier (P0000–P0499) | `P0001` |
| `age` | int | Age in years | `46` |
| `sex` | string | Biological sex | `M` / `F` |
| `race` | string | Race/ethnicity | `White`, `Black`, `Asian`, `Hispanic`, `Other` |
| `chronic_conditions_count` | int | Count of chronic conditions | `5` |

### 2. diagnoses.csv (~1,500 rows)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | string | FK to patients | `P0001` |
| `icd_code` | string | ICD-10 style code | `ICD-675` |
| `diagnosis_name` | string | Clinical name | `Diabetes`, `COPD`, `CKD`, `CHF`, `Hypertension`, `Depression` |
| `diagnosis_date` | date | Date recorded | `2024-07-26` |

### 3. labs.csv (~1,500 rows)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | string | FK to patients | `P0001` |
| `lab_name` | string | Test name | `Creatinine`, `A1c`, `Hemoglobin`, `Glucose`, `BNP`, `WBC`, `Sodium` |
| `lab_value` | float | Numeric result | `1.0`, `10.7`, `9.0` |
| `lab_unit` | string | Unit of measure | `mg/dL`, `%`, `g/dL`, `pg/mL`, `K/uL`, `mEq/L` |
| `abnormal_flag` | bool | Outside reference range | `True` / `False` |
| `result_date` | date | Test date | `2024-08-09` |

**Key Lab Tests:**
- **Creatinine** (mg/dL) — Renal function
- **A1c / HbA1c** (%) — Diabetes control
- **Glucose** (mg/dL) — Blood sugar
- **Hemoglobin** (g/dL) — Anemia screening
- **BNP** (pg/mL) — Heart failure marker
- **WBC** (K/uL) — Infection/inflammation
- **Sodium** (mEq/L) — Electrolyte balance

### 4. medications.csv (~500 rows)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | string | FK to patients | `P0001` |
| `medication_name` | string | Drug name | `Metformin`, `Lisinopril`, `Albuterol`, `Furosemide`, `Metoprolol`, `Sertraline` |
| `class` | string | Drug class | `Antidiabetic`, `Antihypertensive`, `Bronchodilator`, `Diuretic`, `Beta Blocker`, `Antidepressant` |
| `start_date` | date | Prescription start | `2022-10-31` |
| `active_flag` | int | 1=active, 0=inactive | `1` |

### 5. encounters.csv (~1,500 rows)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | string | FK to patients | `P0001` |
| `encounter_type` | string | `Inpatient`, `Outpatient`, `Emergency` | `Inpatient` |
| `admit_date` | date | Admission date | `2024-07-26` |
| `discharge_date` | date | Discharge date | `2024-08-02` |
| `readmission_30d` | int | 1=readmitted within 30 days | `0` / `1` |

### 6. clinical_notes.csv (~1,000 rows)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | string | FK to patients | `P0001` |
| `note_type` | string | `Discharge Summary`, `Progress Note` | `Discharge Summary` |
| `note_text` | string | Free-text clinical note | `Patient is a 46yo M presenting with history of Diabetes, COPD...` |
| `note_date` | date | Note date | `2024-08-02` |

---

## Supported Query Categories

### A. Demographics & Overview
| Query Pattern | Example |
|--------------|---------|
| Patient demographics | `"Patient P0001 age sex race chronic conditions"` |
| Database counts | `"How many patients are above age 65?"`, `"How many female patients?"` |

### B. Diagnoses
| Query Pattern | Example |
|--------------|---------|
| All diagnoses with dates | `"What diagnoses has patient P0001 had, and when?"` |

### C. Clinical Notes / Discharge Summaries
| Query Pattern | Example |
|--------------|---------|
| Discharge summaries | `"Patient P0001 discharge summary notes conditions mentioned"` |
| High readmission risk labels | `"Has patient P0001 ever been labeled high risk for readmission?"` |

### D. Lab Results
| Query Pattern | Example |
|--------------|---------|
| Latest/recent labs | `"Patient P0001 latest lab results abnormal flags"` |
| Specific tests (creatinine, BNP, A1c, glucose, sodium, WBC, hemoglobin) | `"Patient P0001 creatinine BNP A1c glucose sodium WBC hemoglobin"` |
| A1c trend + diabetes control | `"Patient P0001 A1c trend diabetes control worsening improving"` |
| Repeated renal dysfunction | `"Patient P0001 repeated elevated creatinine renal dysfunction"` |
| Infection/inflammation (WBC) | `"Patient P0001 abnormal WBC infection inflammation"` |
| Anemia (hemoglobin) | `"Patient P0001 hemoglobin trend anemia"` |

### E. Condition-Specific Evidence
| Query Pattern | Example |
|--------------|---------|
| Heart failure (CHF dx + BNP + notes) | `"Patient P0001 heart failure CHF BNP evidence"` |
| CKD (diagnosis + creatinine) | `"Patient P0001 chronic kidney disease CKD creatinine"` |
| Diabetes control (dx + glucose + A1c) | `"Patient P0001 diabetes control glucose A1c"` |
| Multimorbidity | `"Patient P0001 multimorbidity multiple chronic conditions"` |

### F. Utilization & Timeline
| Query Pattern | Example |
|--------------|---------|
| Clinical timeline (all events) | `"Patient P0001 timeline what happened recent clinical"` |
| Encounter count + most recent | `"Patient P0001 encounters how many most recent"` |
| Medication-condition alignment | `"Patient P0001 medications align chronic conditions"` |
| 30-day post-discharge follow-up | `"Patient P0001 follow-up 30 day discharge labs encounters meds"` |

### G. Readmission Risk & Handoff
| Query Pattern | Example |
|--------------|---------|
| Readmission risk factors | `"Patient P0001 readmission risk factors"` |
| Clinical handoff summary | `"Patient P0001 handoff summary"` / `"Summarize patient P0001"` |

---

## Example Queries by Patient

```bash
# Demographics
python query.py "Patient P0001 age sex race chronic conditions"

# Diagnoses
python query.py "What diagnoses has patient P0001 had and when"

# Labs
python query.py "Patient P0001 latest lab results abnormal"
python query.py "Patient P0001 A1c trend diabetes control"
python query.py "Patient P0001 creatinine renal dysfunction"
python query.py "Patient P0001 WBC infection inflammation"
python query.py "Patient P0001 hemoglobin anemia"

# Conditions
python query.py "Patient P0001 heart failure CHF BNP"
python query.py "Patient P0001 CKD chronic kidney disease creatinine"
python query.py "Patient P0001 diabetes control glucose A1c"
python query.py "Patient P0001 multimorbidity"

# Notes
python query.py "Patient P0001 discharge summary"
python query.py "Patient P0001 high risk readmission label"

# Timeline & Utilization
python query.py "Patient P0001 timeline recent clinical"
python query.py "Patient P0001 encounters how many most recent"
python query.py "Patient P0001 medications align chronic conditions"
python query.py "Patient P0001 follow-up 30 day discharge"

# Risk & Handoff
python query.py "Patient P0001 readmission risk factors"
python query.py "Patient P0001 handoff summary"

# Population counts
python query.py "How many patients above age 65"
python query.py "How many female patients"
```

---

## Architecture

```
Query → Orchestrator.process_query()
  ├── _is_count_query() → _handle_count_query()          [Population stats]
  ├── _is_complex_clinical_query() → _handle_complex_clinical_query()
  │     └── ClinicalAnalyzer.answer_patient_questions()  [All 6 CSVs joined]
  │           ├── get_patient()
  │           ├── get_diagnoses()
  │           ├── get_labs()
  │           ├── get_medications()
  │           ├── get_encounters()
  │           ├── get_clinical_notes()
  │           └── analysis: readmission, diabetes, renal, HF, anemia, infection, multimorbidity, timeline, followup
  └── Fallback: Planner → Retriever (FAISS) → Evaluator  [General RAG]
```

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Single query
python query.py "Patient P0001 age sex race chronic conditions"

# Interactive (streamlit)
streamlit run streamlit_app.py
```

---

## Fine-Tuning Dataset
Generated IFT dataset: `ift_dataset.jsonl` / `ift_dataset_alpaca.json` (900 examples × 18 question types × 50 patients)

Register with LLaMA-Factory:
```json
"clinical_rag_ift": {
  "file_name": "ift_dataset_alpaca.json",
  "formatting": "alpaca",
  "columns": {"instruction": "instruction", "input": "input", "output": "output"}
}
```