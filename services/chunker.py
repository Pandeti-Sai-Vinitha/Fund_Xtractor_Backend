import fitz
import json
import os
import pickle
import re
from typing import List, Dict, Any


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 500) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def convert_toc_list_to_hierarchy(toc_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    hierarchy = {}
    for entry in toc_entries:
        section = entry["title"]
        page = int(entry["page"])
        if "subsections" in entry:
            hierarchy[section] = {
                "subsections": {
                    sub["title"]: {
                        "pdf_start_page": int(sub["page"]),
                        "pdf_end_page": int(sub["page"])
                    } for sub in entry["subsections"]
                }
            }
        else:
            hierarchy[section] = {
                "pdf_start_page": page,
                "pdf_end_page": page
            }
    return hierarchy


def is_dot_heavy(text: str) -> bool:
    dot_lines = sum(1 for line in text.splitlines() if re.search(r"(\s*\.\s*){5,}", line))
    total_lines = len(text.splitlines())
    return total_lines > 0 and dot_lines / total_lines > 0.5


def clean_text(text: str) -> str:
    text = re.sub(r"(\s*\.\s*){5,}", " ", text)  # Remove long dot sequences
    text = re.sub(r"\.{3,}", " ", text)          # Replace 3+ dots with space
    return text


def chunk_pdf_by_toc(
    pdf_path: str,
    toc_json_path: str,
    output_pkl_path: str,
    chunk_size: int = 2000,
    overlap: int = 500
) -> List[Dict[str, Any]]:

    with open(toc_json_path, "r", encoding="utf-8") as f:
        toc_entries = json.load(f)

# ✅ Unwrap if wrapped under "hierarchy"
    if isinstance(toc_entries, dict) and "hierarchy" in toc_entries:
        toc_entries = toc_entries["hierarchy"]

    if isinstance(toc_entries, list):
        hierarchy = convert_toc_list_to_hierarchy(toc_entries)
    elif isinstance(toc_entries, dict):
        hierarchy = toc_entries
    else:
        raise ValueError("Unsupported TOC format: must be a list or dict")


    doc = fitz.open(pdf_path)
    all_chunks = []

    def process_chunk(section: str, subsection: str, subdata: Dict[str, Any]):
        if "pdf_start_page" not in subdata or "pdf_end_page" not in subdata:
            print(f"❌ Skipping {section} → {subsection or 'FULL'}: missing page range")
            return

        start_page = subdata["pdf_start_page"]
        end_page = subdata["pdf_end_page"]
        text = ""

        for i in range(start_page, end_page + 1):
            if 1 <= i <= len(doc):
                page_text = doc[i - 1].get_text()
                if not page_text.strip():
                    print(f"⚠️ Page {i} in {section} → {subsection or 'FULL'} has no extractable text")
                text += page_text
            else:
                print(f"⏭️ Skipping page {i}: out of bounds (doc has {len(doc)} pages)")

        if not text.strip():
            print(f"⚠️ Skipping {section} → {subsection or 'FULL'}: no extractable text in range {start_page}–{end_page}")
            return

        if is_dot_heavy(text):
            print(f"⏭️ Skipping dot-heavy section: {section} ({start_page}–{end_page})")
            return

        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            chunk_meta = {
            "chunk_id": f"{section}__{subsection or 'FULL'}__{idx}",
            "section": section,
            "subsection": subsection or "FULL",
            "chunk_index": idx,
            "start_page": subdata.get("start_page"),
            "end_page": subdata.get("end_page"),
            "pdf_start_page": start_page,
            "pdf_end_page": end_page,
            "char_count": len(chunk),
            "text": chunk,
            "source": os.path.basename(pdf_path),
        }
        all_chunks.append(chunk_meta)


    for section, data in hierarchy.items():
        subsections = data.get("subsections", {})
        if subsections:
            for subsection, subdata in subsections.items():
                process_chunk(section, subsection, subdata)
        else:
            process_chunk(section, None, data)

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Saved {len(all_chunks)} chunks to {output_pkl_path} with overlap={overlap}")
    return all_chunks
