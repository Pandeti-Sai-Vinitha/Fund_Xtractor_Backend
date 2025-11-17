import os
import datetime
import numpy as np
import requests
import textwrap
import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b) -> float:
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    return float(np.dot(a, b))


def get_top_k_chunks_semantic(
    question: str,
    chunks: List[Dict[str, Any]],
    k: int = 10,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[List[Dict[str, Any]], str, str]:

    model = SentenceTransformer(model_name)
    query_vector = model.encode([question])[0]

    scored_chunks = []
    for chunk in chunks:
        embedding = chunk.get("embedding_vector")
        if embedding is None or not isinstance(embedding, (list, np.ndarray)):
            continue
        try:
            score = cosine_similarity(query_vector, embedding)
            chunk_with_score = chunk.copy()
            chunk_with_score["similarity"] = score
            scored_chunks.append(chunk_with_score)
        except Exception as e:
            print(f"❌ Skipping chunk due to similarity error: {e}")
            continue

    top_chunks = sorted(scored_chunks, key=lambda x: x["similarity"], reverse=True)[:k]

    section_counts = Counter(c.get("section", "").strip().upper() for c in top_chunks)
    subsection_counts = Counter(c.get("subsection", "").strip() for c in top_chunks)

    inferred_section = section_counts.most_common(1)[0][0] if section_counts else ""
    inferred_subsection = subsection_counts.most_common(1)[0][0] if subsection_counts else ""

    return top_chunks, inferred_section, inferred_subsection


def build_llm_prompt_for_risk_summary(
    user_question: str,
    search_intent: str,
    top_chunks: List[Dict[str, Any]],
    remarks: str = "",
    section: str = "",
    subsection: str = ""
) -> str:

    chunk_texts = "\n\n".join(
        f"Chunk {i+1}:\n{textwrap.indent(chunk['text'].strip(), '  ')}"
        for i, chunk in enumerate(top_chunks)
    )

    prompt = f"""
You are a risk analyst reviewing the company's DHRP. Based on the chunks below, generate a structured risk summary.

Explicit Scope:
Section: {section}
Subsection: {subsection}

AI searched for:
{search_intent}

Chunks:
{chunk_texts}

Task:
1. Identify all risk factors mentioned.
2. Group them under Internal and External risks.
3. Within each group, organize risks by category (e.g. Financial, Operational, Regulatory, Competitive).
4. For each risk, briefly explain severity, impact, mitigation, and any gaps.
5. Use this format:

Internal Risks:
- [Category: Financial]
  - Risk: ...
    - Severity: ...
    - Impact: ...
    - Mitigation: ...
    - Gaps: ...

External Risks:
- [Category: Regulatory]
  - Risk: ...
    - Severity: ...
    - Impact: ...
    - Mitigation: ...
    - Gaps: ...

Remarks:
{remarks or 'None'}

Present the summary in a clear, structured format suitable for compliance review.
""".strip()

    return prompt


def parse_structured_risk_summary(text: str) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    result = defaultdict(lambda: defaultdict(list))
    current_group = None
    current_category = None
    current_risk = {}

    def flush_risk():
        nonlocal current_risk, current_group, current_category
        if current_risk and current_group and current_category:
            result[current_group][current_category].append(current_risk)
        current_risk = {}

    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        line = line.replace("**", "").strip()

        if re.match(r"internal risks", line, re.IGNORECASE):
            flush_risk()
            current_group = "Internal"
            current_category = None
        elif re.match(r"external risks", line, re.IGNORECASE):
            flush_risk()
            current_group = "External"
            current_category = None
        elif re.match(r"-\s*\[Category:\s*(.*?)\]", line):
            flush_risk()
            match = re.match(r"-\s*\[Category:\s*(.*?)\]", line)
            if match:
                current_category = match.group(1).strip()
        elif re.match(r"-\s*Risk\s*:\s*", line, re.IGNORECASE):
            flush_risk()
            current_risk = {"Risk": line.split(":", 1)[1].strip()}
        elif line.startswith("-") and ":" in line and current_risk is not None:
            key, val = line[1:].split(":", 1)
            current_risk[key.strip()] = val.strip()

    flush_risk()
    return result


def evaluate_with_gemini(prompt: str, cache_path: str) -> Tuple[str, Dict[str, Dict[str, List[Dict[str, str]]]]]:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": "AIzaSyAXkiytsobQRY8OfPEonFBhNcx-yyCooLI"
    }
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            summary = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(summary)

            parsed = parse_structured_risk_summary(summary)
            json_path = cache_path.replace(".txt", ".json")

            if parsed and any(parsed.values()):
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(parsed, jf, indent=2)
            else:
                print("⚠️ Parsed summary is empty. Falling back to raw text only.")

            return summary, parsed
        except Exception as e:
            print("Failed to parse Gemini response:", response.text)
            return f"[JSON parsing error: {str(e)}]", {}
    else:
        print(f"Gemini API error {response.status_code}: {response.text}")
        return f"[API Error {response.status_code}]", {}


def generate_risk_summary_from_chunks(
    chunks: List[Dict[str, Any]],
    user_question: str = "What are the key risk factors disclosed in the DHRP?",
    search_intent: str = (
        "Search 'Risk Factors', 'Business Overview', and "
        "'Management Discussion and Analysis' for risk-related disclosures."
    ),
    remarks: str = (
        "Focus on financial, operational, regulatory, and competitive risks. "
        "Include any mitigation strategies or gaps."
    ),
    cache_dir: str = "cache"
) -> Tuple[str, Dict[str, Dict[str, List[Dict[str, str]]]], List[Dict[str, Any]]]:

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_path = os.path.join(cache_dir, f"risk_summary_{timestamp}.txt")

    top_k_chunks, inferred_section, inferred_subsection = get_top_k_chunks_semantic(
        user_question, chunks, k=10
    )

    prompt = build_llm_prompt_for_risk_summary(
        user_question,
        search_intent,
        top_k_chunks,
        remarks,
        section=inferred_section,
        subsection=inferred_subsection,
    )

    summary, parsed_summary = evaluate_with_gemini(prompt, cache_path)

    return summary, parsed_summary, top_k_chunks