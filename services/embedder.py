import pickle
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any

def embed_chunks_optimal(
    input_pkl_path: str,
    output_pkl_path: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict[str, Any]]:
    with open(input_pkl_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"📦 Loaded {len(chunks)} chunks from {input_pkl_path}")

    model = SentenceTransformer(model_name)

    embedding_inputs = []
    valid_indices = []

    for i, chunk in enumerate(chunks):
        section = chunk.get("section", "")
        subsection = chunk.get("subsection", "")
        text = chunk.get("text", "")
        if text.strip():
            embedding_input = f"Section: {section}\nSubsection: {subsection}\nContent: {text}"
            chunk["embedding_input"] = embedding_input
            embedding_inputs.append(embedding_input)
            valid_indices.append(i)
        else:
            chunk["embedding_input"] = ""
            chunk["embedding_vector"] = None

    if not embedding_inputs:
        print("⚠️ No valid chunks found for embedding.")
        return chunks

    print("⚙️ Generating embeddings...")
    embeddings = model.encode(embedding_inputs, show_progress_bar=True)

    for j, i in enumerate(valid_indices):
        chunks[i]["embedding_vector"] = embeddings[j]

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Saved embedded chunks to {output_pkl_path}")
    return chunks
