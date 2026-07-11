import os
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class RetrievalAgent:
    def __init__(self, index_path: str = str(DATA_DIR / "faiss_index.bin"), meta_path: str = str(DATA_DIR / "chunks_meta.pkl")):
        self.index_path = index_path
        self.meta_path = meta_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = None
            self.metadata = []

    def retrieve_context(self, plan: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []

        search_text = plan.rewritten_query
        query_vector = self.encoder.encode([search_text]).astype('float32')

        # Retrieve more than top_k to allow for filtering headroom
        search_k = top_k * 20 if plan.patient_id_filter else top_k
        distances, indices = self.index.search(query_vector, search_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                meta = self.metadata[idx]
                
                # Apply Metadata Filter for Patient ID
                if plan.patient_id_filter and meta.get('patient_id') != plan.patient_id_filter:
                    continue

                results.append({
                    "chunk_id": int(idx),
                    "text": meta['text'],
                    "metadata": {
                        "patient_id": meta.get('patient_id', 'Unknown'),
                        "source": meta.get('source_table', 'Unknown')
                    },
                    "score": float(distances[0][i])
                })
                
                if len(results) >= top_k:
                    break
        return results
