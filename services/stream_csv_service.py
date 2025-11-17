import json
import logging
import os

import pandas as pd


from services.core_logic import build_llm_prompt_for_fund_setup, evaluate_with_gemini, get_chunks_with_fallback, parse_gemini_response
from utils.helpers import update_processing_stage


def stream_csv_evaluation(
    base: str,
    chunks: list,
    csv_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2"
):
    status_path = os.path.join('status', f"{base}.json")
    os.makedirs('status', exist_ok=True)
    status_data = {
        "answers": {},
        "reasoningSteps": {},
        "milestones": {},
        "done": False
    }

    def save_status():
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    answers = []

    for idx, row in df.iterrows():
        data_field = str(row.get("Data Fields", "")).strip()
        particulars = str(row.get("Particulars", "")).strip()
        search_intent = str(row.get("Where we get Information in Prospectus", "")).strip()
        section = str(row.get("Heading", "")).strip()
        subsection = str(row.get("Sub- Heading", "")).strip()

        user_question = particulars if particulars and particulars.upper() != "N/A" else data_field

        milestone = f"🔍 Q{idx+1}: {data_field} — Searching relevant content..."
        update_processing_stage(base, milestone)
        status_data["milestones"].setdefault(data_field, []).append(milestone)
        save_status()

        top_k_chunks, final_section, final_subsection = get_chunks_with_fallback(
            user_question, chunks, k=10, model_name=model_name,
            section=section, subsection=subsection
        )

        milestone = f"🎯 Q{idx+1}: Selecting best matches..."
        update_processing_stage(base, milestone)
        status_data["milestones"][data_field].append(milestone)
        save_status()

        prompt = build_llm_prompt_for_fund_setup(
            data_field, particulars, search_intent, top_k_chunks,
            final_section, final_subsection
        )

        milestone = f"🧠 Q{idx+1}: Generating answer..."
        update_processing_stage(base, milestone)
        status_data["milestones"][data_field].append(milestone)
        save_status()

        cache_path = f"cache/gemini_summary_row_{idx+1}.txt"
        summary = evaluate_with_gemini(prompt, cache_path)
        parsed = parse_gemini_response(summary)

        milestone = f"✅ Q{idx+1}: Completed."
        update_processing_stage(base, milestone)
        status_data["milestones"][data_field].append(milestone)
        save_status()

        for step in parsed.get("reasoning_steps", []):
            reasoning = f"🧩 {step}"
            update_processing_stage(base, reasoning)
            status_data["reasoningSteps"].setdefault(data_field, []).append(step)
            save_status()

        status_data["answers"][data_field] = parsed["answer"]
        save_status()

        answers.append(parsed["answer"])

    df["Answer"] = answers
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ CSV evaluation complete: {output_path}")

    status_data["done"] = True
    save_status()