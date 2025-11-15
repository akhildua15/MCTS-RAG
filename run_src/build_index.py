#!/usr/bin/env python3
import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse

def build_faiss_for_dataset(dataset_name, file_name):
    dataset_dir = f"data/{dataset_name}"
    evidence_json_path = f"{dataset_dir}/{file_name}.json"
    evidence_csv_path = f"{dataset_dir}/evidence.csv"
    faiss_index_path = f"{dataset_dir}/faiss.index"

    if not os.path.exists(evidence_json_path):
        raise FileNotFoundError(f"{evidence_json_path} does not exist.")

    print(f"[INFO] Loading evidence from {evidence_json_path}")
    with open(evidence_json_path, "r") as f:
        evidences = json.load(f)

    # Convert to CSV
    rows = []
    for ev in evidences:
        if "evidence" in ev:
            # If evidence is a list-of-dicts (like FMT)
            rows.append(ev["evidence"])
        else:
            # If flat
            rows.append(ev)

    df = pd.DataFrame(rows, columns=["evidence"])
    df.to_csv(evidence_csv_path, index=False)
    print(f"[INFO] Saved CSV to {evidence_csv_path}")

    # Build embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(df["evidence"].tolist(), convert_to_numpy=True)
    emb = emb.astype("float32")

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, faiss_index_path)

    print(f"[INFO] Saved FAISS index to {faiss_index_path}")
    print(f"[INFO] Completed FAISS build for dataset: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    build_faiss_for_dataset(args.dataset, args.file)