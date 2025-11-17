import fitz
import re
import os
import json
from typing import Dict, Any
 
 
def extract_footer_page_map(doc: fitz.Document) -> Dict[int, int]:
    """
    Extracts a mapping from PDF page index to footer page number.
    Example: {0: 1, 1: 2, ...}
    """
    page_map = {}
    for i in range(len(doc)):
        bottom_text = doc[i].get_text("text", clip=fitz.Rect(0, 700, 595, 842))
        for line in bottom_text.splitlines():
            match = (
                re.match(r"^Page\s*(\d+)$", line.strip(), re.IGNORECASE)
                or re.match(r"^(\d+)$", line.strip())
            )
            if match:
                footer_page = int(match.group(1))
                page_map[i] = footer_page
                break
    return page_map
 
 
def extract_toc_sections_hierarchical(
    pdf_path: str,
    max_pages: int = 10,
    output_json_path: str = None,
    output_txt_path: str = None,
) -> Dict[str, Any]:
    """
    Extracts a hierarchical TOC from the first `max_pages` of a PDF.
    Returns a dictionary of sections and subsections with page ranges.
    Optionally saves JSON and TXT outputs.
    """
    doc = fitz.open(pdf_path)
    page_map = extract_footer_page_map(doc)
    reverse_map = {v: k for k, v in page_map.items()}
 
    toc_text = ""
    for i in range(min(max_pages, len(doc))):
        toc_text += doc[i].get_text()
 
    lines = toc_text.splitlines()
    flat_map = {}
 
    pattern = re.compile(r"^(.*?)\s*(?:\.{2,}|\s{4,})\s*(\d+)$")
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            title = match.group(1).strip()
            page = int(match.group(2))
            if title:
                flat_map[title] = page
 
    # Compute offset using first TOC page if available
    first_toc_page = min(flat_map.values()) if flat_map else 1
    offset = reverse_map.get(first_toc_page, 0) - first_toc_page
 
    keys = list(flat_map.keys())
    values = list(flat_map.values())
    hierarchy = {}
    current_section = None
 
    for i, title in enumerate(keys):
        start = values[i]
        end = values[i + 1] - 1 if i + 1 < len(values) else start
        pdf_start = start + offset
        pdf_end = end + offset
 
        # Normalize reversed ranges
        if start > end:
            start, end = end, start
        if pdf_start > pdf_end:
            pdf_start, pdf_end = pdf_end, pdf_start
 
        if title.upper().startswith("SECTION"):
            current_section = title
            hierarchy[current_section] = {
                "start_page": start,
                "end_page": end,
                "pdf_start_page": pdf_start,
                "pdf_end_page": pdf_end,
                "subsections": {},
            }
        else:
            if current_section:
                hierarchy[current_section]["subsections"][title] = {
                    "start_page": start,
                    "end_page": end,
                    "pdf_start_page": pdf_start,
                    "pdf_end_page": pdf_end,
                }
            else:
                hierarchy[title] = {
                    "start_page": start,
                    "end_page": end,
                    "pdf_start_page": pdf_start,
                    "pdf_end_page": pdf_end,
                }
 
    # Save outputs if paths provided
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2)
 
    if output_txt_path:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for section, data in hierarchy.items():
                f.write(
                    f"{section} : Pages {data['start_page']}–{data['end_page']} "
                    f"(PDF {data['pdf_start_page']}–{data['pdf_end_page']})\n"
                )
                for sub, subdata in data.get("subsections", {}).items():
                    f.write(
                        f"  - {sub} : Pages {subdata['start_page']}–{subdata['end_page']} "
                        f"(PDF {subdata['pdf_start_page']}–{subdata['pdf_end_page']})\n"
                    )
 
    return hierarchy