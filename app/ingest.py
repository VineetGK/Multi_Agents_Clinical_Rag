
import pandas as pd
import faiss
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")

def build_index():
    print("Loading synthetic data...")
    patients = pd.read_csv(f'{DATA_DIR}/patients.csv')
    encounters = pd.read_csv(f'{DATA_DIR}/encounters.csv')
    labs = pd.read_csv(f'{DATA_DIR}/labs.csv')
    meds = pd.read_csv(f'{DATA_DIR}/medications.csv')
    diagnoses = pd.read_csv(f'{DATA_DIR}/diagnoses.csv')
    notes = pd.read_csv(f'{DATA_DIR}/clinical_notes.csv')

    chunks = []
    metadata = []

    print("Creating text chunks...")
    # Process Notes
    for _, row in notes.iterrows():
        text = f"[Patient: {row['patient_id']}] Clinical Note [{row['note_type']}] on {row['note_date']}: {row['note_text']}"
        chunks.append(text)
        metadata.append({'patient_id': row['patient_id'], 'source_table': 'clinical_notes'})

    # Process Labs
    for _, row in labs.iterrows():
        abnormal = "ABNORMAL" if row['abnormal_flag'] else "NORMAL"
        text = f"[Patient: {row['patient_id']}] Lab Result on {row['result_date']}: {row['lab_name']} = {row['lab_value']} {row['lab_unit']} ({abnormal})"
        chunks.append(text)
        metadata.append({'patient_id': row['patient_id'], 'source_table': 'labs'})

    # Process Medications
    for _, row in meds.iterrows():
        status = "Active" if row['active_flag'] else "Inactive"
        text = f"[Patient: {row['patient_id']}] Medication started {row['start_date']}: {row['medication_name']} ({row['class']}) - Status: {status}"
        chunks.append(text)
        metadata.append({'patient_id': row['patient_id'], 'source_table': 'medications'})

    # Process Diagnoses
    for _, row in diagnoses.iterrows():
        text = f"[Patient: {row['patient_id']}] Diagnosis on {row['diagnosis_date']}: {row['diagnosis_name']} (ICD: {row['icd_code']})"
        chunks.append(text)
        metadata.append({'patient_id': row['patient_id'], 'source_table': 'diagnoses'})

    # Process Encounters
    for _, row in encounters.iterrows():
        readm = "High risk for 30-day readmission" if row['readmission_30d'] else "No 30-day readmission risk"
        text = f"[Patient: {row['patient_id']}] Encounter ({row['encounter_type']}) from {row['admit_date']} to {row['discharge_date']}. {readm}."
        chunks.append(text)
        metadata.append({'patient_id': row['patient_id'], 'source_table': 'encounters'})

    print(f"Total chunks created: {len(chunks)}")

    print("Generating embeddings (this may take a minute)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = encoder.encode(chunks, show_progress_bar=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    # Save the index and metadata
    faiss.write_index(index, f'{DATA_DIR}/faiss_index.bin')

    # Add text to metadata for retrieval
    for m, text in zip(metadata, chunks):
        m['text'] = text

    with open(f'{DATA_DIR}/chunks_meta.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("Index and metadata saved successfully!")

if __name__ == '__main__':
    build_index()
