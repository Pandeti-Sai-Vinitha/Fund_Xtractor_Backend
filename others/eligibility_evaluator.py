import os
import re
import textwrap
import requests
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b) -> float:
    
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    return float(np.dot(a, b))


def match_subsection_regex(target_subsection: str, chunks: list, section: str = None):
    
    if not target_subsection:
        return []

    pattern = re.compile(re.escape(target_subsection.strip()), re.IGNORECASE)
    matched = []

    for chunk in chunks:
        sec = chunk.get("section", "").strip().upper()
        sub = chunk.get("subsection", "").strip()

        if section and sec != section.strip().upper():
            continue

        if pattern.search(sub):
            matched.append(chunk)

    return matched


def score_and_return_top_k(query_vector, chunk_list, k: int):
    
    scored = []
    for chunk in chunk_list:
        if "embedding_vector" not in chunk:
            continue
        score = cosine_similarity(query_vector, chunk["embedding_vector"])
        chunk_with_score = chunk.copy()
        chunk_with_score["similarity"] = score
        scored.append(chunk_with_score)
    return sorted(scored, key=lambda x: x["similarity"], reverse=True)[:k]


def get_chunks_with_fallback(
    question: str,
    chunks: list,
    k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    section: str = None,
    subsection: str = None
) -> Tuple[List[Dict[str, Any]], str, str]:
    
    model = SentenceTransformer(model_name)
    query_vector = model.encode([question])[0]

    section = section.strip().upper() if section else None
    subsection = subsection.strip() if subsection else None

    
    exact_filtered = [
        c for c in chunks
        if (section is None or c.get("section", "").strip().upper() == section)
        and (subsection is None or c.get("subsection", "").strip() == subsection)
    ]
    if exact_filtered:
        return score_and_return_top_k(query_vector, exact_filtered, k), section or "", subsection or ""

    
    regex_filtered = match_subsection_regex(subsection, chunks, section=section)
    if regex_filtered:
        return score_and_return_top_k(query_vector, regex_filtered, k), section or "", subsection or ""

    
    scored = []
    for chunk in chunks:
        if "embedding_vector" not in chunk:
            continue
        score = cosine_similarity(query_vector, chunk["embedding_vector"])
        chunk_with_score = chunk.copy()
        chunk_with_score["similarity"] = score
        scored.append(chunk_with_score)

    top_k = sorted(scored, key=lambda x: x["similarity"], reverse=True)[:k]

    section_counts = Counter(c.get("section", "").strip().upper() for c in top_k)
    subsection_counts = Counter(c.get("subsection", "").strip() for c in top_k)

    inferred_section = section_counts.most_common(1)[0][0] if section_counts else ""
    inferred_subsection = subsection_counts.most_common(1)[0][0] if subsection_counts else ""

    return top_k, inferred_section, inferred_subsection


import textwrap
from typing import List, Dict, Any

def build_llm_prompt_for_fund_setup(
    data_field: str,
    particulars: str,
    search_intent: str,
    top_chunks: List[Dict[str, Any]],
    section: str = "",
    subsection: str = ""
) -> str:
    
    # Combine all chunk texts into a single block
    combined_text = "\n\n".join(chunk["text"].strip() for chunk in top_chunks)

    # Choose task instruction based on search_intent
    if search_intent:
        task_instruction = (
            f"Use the guidance in 'Where to find information' to locate and extract the answer for '{data_field}'. "
            f"Respond in a single, clear sentence unless the answer requires explanation. Do not mention how the information was retrieved or structured."
        )
    else:
        task_instruction = (
            f"Extract the answer for '{data_field}' from the provided content. "
            f"Respond in a single, clear sentence unless the answer requires explanation. Avoid referring to document structure or retrieval method."
        )

    # Final prompt
    prompt = f"""
You are extracting fund setup information from a prospectus document.

Section: {section}
Subsection: {subsection}

Data Field: {data_field}
Question or Particulars: {particulars}
Where to find information: {search_intent}

Document Content:
{combined_text}

Instruction:
{task_instruction}
""".strip()

    return prompt


    


def evaluate_with_gemini(prompt: str, cache_path: str) -> str:
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": "AIzaSyD23Jo4P4hqJZJFRiuIAt1W7-3WvmF3w-c"
    }
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=payload)
    summary_filename = os.path.basename(cache_path)

    if response.status_code == 200:
        try:
            summary = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f" Summary saved to: {summary_filename}")
            return summary
        except Exception as e:
            print(" Failed to parse Gemini response:", response.text)
            return f"[JSON parsing error: {str(e)}]"
    else:
        print(f" Gemini API error {response.status_code}: {response.text}")
        return f"[API Error {response.status_code}]"


def process_csv_and_evaluate(
    csv_path: str,
    output_path: str,
    chunks: list,
    model_name: str = "all-MiniLM-L6-v2"
) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    results = []
    for idx, row in df.iterrows():
        data_field = str(row.get("Data Fields", "")).strip()
        particulars = str(row.get("Particulars", "")).strip()
        search_intent = str(row.get("Where we get Information in Prospectus", "")).strip()
        section = str(row.get("Heading", "")).strip()
        subsection = str(row.get("Sub- Heading", "")).strip()

        # Use particulars as the search query, fallback to data_field if particulars is empty or "N/A"
        if not particulars or particulars.upper() == "N/A":
            user_question = data_field if data_field else f"Information about {data_field}"
        else:
            user_question = particulars

        print(f"\n Processing row {idx+1}: {data_field} - {particulars[:50]}...")

        top_k_chunks, final_section, final_subsection = get_chunks_with_fallback(
            user_question, chunks, k=10, model_name=model_name,
            section=section, subsection=subsection
        )

        prompt = build_llm_prompt_for_fund_setup(
            data_field, particulars, search_intent, top_k_chunks,
            final_section, final_subsection
        )

        cache_path = f"cache/gemini_summary_row_{idx+1}.txt"
        summary = evaluate_with_gemini(prompt, cache_path)

        results.append(summary)

    df["Answer"] = results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n All rows processed. Output saved to: {output_path}")
    return df
