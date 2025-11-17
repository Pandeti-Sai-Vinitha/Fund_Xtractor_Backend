import os, logging, pickle, requests, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer

from services.core_logic import cosine_similarity
from utils.helpers import normalize_name, get_base_name
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def analyze_csv_file(selected_doc, csv_file):
    if not selected_doc or not csv_file or csv_file.filename == '':
        logging.warning("⚠️ CSV analysis failed — Missing document or CSV file")
        return False, {"error": "Missing document or CSV file"}

    base = normalize_name(get_base_name(selected_doc))
    pkl_path = f"pickles/{base}.pkl"
    if not os.path.exists(pkl_path):
        logging.warning(f"⚠️ Pickle file not found for: {selected_doc}")
        return False, {"error": f"Pickle file not found for {selected_doc}"}

    with open(pkl_path, 'rb') as f:
        chunks = pickle.load(f)

    df = pd.read_csv(csv_file, quotechar='"')
    logging.info(f"📥 Analyzing CSV for: {selected_doc} — Questions: {len(df)}")

    answers = []
    for _, row in df.iterrows():
        question = row['Question']
        top_chunks = get_top_chunks(question, chunks, top_k=3)
        combined_answer = ""
        for chunk_text in top_chunks:
            prompt = (
                f"Answer the following question briefly and precisely based on this section of a DHRP document. "
                f"Limit your response to 2–3 sentences max.\n\n"
                f"Question: {question}\n\n"
                f"Section:\n{chunk_text[:3000]}"
            )
            payload = { "contents": [{"parts": [{"text": prompt}]}] }
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                try:
                    answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                    combined_answer += answer.strip() + " "
                except Exception as parse_err:
                    logging.error(f"❌ Failed to parse Gemini response for question: {question} — {parse_err}")
                    continue
            else:
                logging.error(f"❌ Gemini API error {response.status_code} — {response.text}")
        answers.append(combined_answer.strip() if combined_answer else "❌ No answer generated")

    df['Answer'] = answers
    output_path = f"answered_csv/{base}_analysis.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"✅ CSV analysis complete — Output saved to: {output_path}")
    return True, {"message": "CSV analyzed successfully", "filename": f"{base}_analysis.csv"}

def view_csv_file(filename):
    csv_path = os.path.join('answered_csv', filename)
    if not os.path.exists(csv_path):
        logging.warning(f"⚠️ CSV file not found for viewing: {filename}")
        return False, {"success": False, "message": f"❌ CSV file not found: {filename}"}

    df = pd.read_csv(csv_path)
    df = df.replace({np.nan: None})
    records = df.to_dict(orient='records')
    columns = list(df.columns)
    base_name = os.path.splitext(filename)[0]

    logging.info(f"📄 CSV viewed: {filename} — Rows: {len(records)}")
    return True, {
        "success": True,
        "filename": filename,
        "base_name": base_name,
        "columns": columns,
        "rows": records
    }


def get_top_chunks(question, chunks, top_k=3):
    chunk_texts = []
    for chunk in chunks:
        if isinstance(chunk, str):
            chunk_texts.append(chunk)
        elif isinstance(chunk, dict) and "text" in chunk:
            chunk_texts.append(chunk["text"])
        else:
            continue
    chunk_embeddings = embedding_model.encode(chunk_texts)
    question_embedding = embedding_model.encode([question])[0]
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunk_texts[i] for i in top_indices]
