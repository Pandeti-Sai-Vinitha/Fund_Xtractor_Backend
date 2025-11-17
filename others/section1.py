import re
import pandas as pd
from PyPDF2 import PdfReader
from typing import List, Tuple, cast

SECTION_DETECT_RE = re.compile(
    r"^\s*SECTION\s+([IVXLCDM]+)\b(.*)$",
    re.IGNORECASE,
)

SECTION_RE = re.compile(
    r"^\s*SECTION\s+([IVXLCDM]+)\s*(?:[:：.\-\u2010-\u2015])?\s*(.+)$",
    re.IGNORECASE,
)

def extract_text_range(pdf_reader, start, end):
    text = ""
    start = max(1, int(start))
    end = min(len(pdf_reader.pages), int(end))
    for i in range(start - 1, end):
        text += pdf_reader.pages[i].extract_text() or ""
    return text

def _split_title_and_page(line: str):
    line = re.sub(r"\.{2,}", " ", line.strip())
    line = re.sub(r"\s{2,}", " ", line)
    m = re.search(r"(\d+)\s*$", line)
    if m:
        page = int(m.group(1))
        title = line[:m.start()].rstrip()
    else:
        page = None
        title = line
    return title, page

def _is_toc_like_page(text: str) -> bool:
    up = text.upper()
    if "TABLE OF CONTENTS" in up:
        return True
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    trailing_nums = sum(1 for ln in lines if re.search(r"\d+\s*$", ln))
    dotted = sum(1 for ln in lines if re.search(r"\.{3,}", ln))
    return trailing_nums >= 5 and dotted >= 3

def _extract_multi_title_page_pairs(line: str) -> List[Tuple[str, int]]:
    cleaned = re.sub(r"\.{2,}", " ", line.strip())
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if SECTION_DETECT_RE.match(cleaned):
        return []
    pairs: List[Tuple[str, int]] = []
    for m in re.finditer(r"([^\d\n][^0-9]*?)\s+(\d+)(?=\s|$)", cleaned):
        title = cast(str, m.group(1)).strip()
        page = int(cast(str, m.group(2)))
        if title:
            pairs.append((title, page))
    return pairs

def _extract_toc_internal(pdf_reader, page_offset=0, max_toc_pages=5):
    toc = []
    current_section = None
    toc_page_number = None
    subsections_buffer = []
    in_toc = False
    pages_seen_since_toc = 0

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text() or ""
        up = text.upper()

        if not in_toc and "TABLE OF CONTENTS" in up:
            in_toc = True
            toc_page_number = page_number

        if not in_toc:
            continue

        if not _is_toc_like_page(text):
            break

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            raw = lines[i]
            title, page_num = _split_title_and_page(raw)

            if title and SECTION_DETECT_RE.match(title):
                pretty_title = title
                m_clean = SECTION_RE.match(title)
                if m_clean:
                    pretty_title = m_clean.group(0)

                if page_num is None:
                    j = i + 1
                    combined_title = pretty_title
                    while j < len(lines) and j <= i + 8:
                        t2, p2 = _split_title_and_page(lines[j])
                        if p2 is not None:
                            combined_title = (combined_title + (" " + t2 if t2 else "")).strip()
                            page_num = p2
                            i = j
                            break
                        elif t2.strip():
                            combined_title += " " + t2
                        j += 1
                    pretty_title = combined_title

                if current_section:
                    if subsections_buffer:
                        current_section["subsections"].append({
                            "subsection": " ".join(subsections_buffer).strip(),
                            "start_page": None,
                            "end_page": None,
                        })
                        subsections_buffer = []
                    toc.append(current_section)

                current_section = {
                    "section": pretty_title.strip(),
                    "subsections": [],
                    "start_page": (page_num + page_offset) if page_num is not None else None,
                    "end_page": None,
                }
                i += 1
                continue

            multi = _extract_multi_title_page_pairs(raw)
            if multi:
                if current_section:
                    if subsections_buffer:
                        buffered = " ".join(subsections_buffer).strip()
                        if buffered:
                            current_section["subsections"].append({
                                "subsection": buffered,
                                "start_page": None,
                                "end_page": None,
                            })
                        subsections_buffer = []
                    for t, p in multi:
                        current_section["subsections"].append({
                            "subsection": t,
                            "start_page": p + page_offset,
                            "end_page": None,
                        })
                else:
                    for t, p in multi:
                        toc.append({
                            "section": t,
                            "subsections": [{
                                "subsection": t,
                                "start_page": p + page_offset,
                                "end_page": None,
                            }],
                            "start_page": p + page_offset,
                            "end_page": None,
                        })
                i += 1
                continue

            if current_section:
                if page_num is None:
                    if title and title.strip():
                        subsections_buffer.append(title)
                else:
                    if subsections_buffer:
                        buffered = " ".join(subsections_buffer).strip()
                        if buffered:
                            current_section["subsections"].append({
                                "subsection": buffered,
                                "start_page": None,
                                "end_page": None,
                            })
                        subsections_buffer = []
                    if title and title.strip():
                        current_section["subsections"].append({
                            "subsection": title.strip(),
                            "start_page": page_num + page_offset,
                            "end_page": None
                        })
            else:
                if page_num is not None and (title and title.strip()):
                    toc.append({
                        "section": title.strip(),
                        "subsections": [{
                            "subsection": title.strip(),
                            "start_page": page_num + page_offset,
                            "end_page": None,
                        }],
                        "start_page": page_num + page_offset,
                        "end_page": None,
                    })

            i += 1

        pages_seen_since_toc += 1
        if pages_seen_since_toc >= max_toc_pages:
            break

        if page_number + 1 <= len(pdf_reader.pages):
            nxt = (pdf_reader.pages[page_number].extract_text() or "")
            if not _is_toc_like_page(nxt):
                break

    if current_section:
        if subsections_buffer:
            current_section["subsections"].append({
                "subsection": " ".join(subsections_buffer).strip(),
                "start_page": None,
                "end_page": None,
            })
        toc.append(current_section)

    for section in toc:
        if not section["subsections"]:
            section["subsections"].append({
                "subsection": section["section"],
                "start_page": section.get("start_page"),
                "end_page": None
            })

    toc = correct_end_pages(toc, len(pdf_reader.pages))
    for section in toc:
        for sub in section["subsections"]:
            if not isinstance(sub.get("start_page"), int):
                sub["start_page"] = section.get("start_page")

    return toc, toc_page_number

def infer_page_offset(pdf_reader, toc):
    candidates = []
    for section in toc:
        s_title = (section.get("section") or "").strip()
        s_start = section.get("start_page")
        if isinstance(s_start, int) and s_title:
            candidates.append((s_title, s_start))
        for sub in section.get("subsections", []):
            t = (sub.get("subsection") or "").strip()
            p = sub.get("start_page")
            if isinstance(p, int) and t:
                candidates.append((t, p))

    candidates.sort(key=lambda x: x[1])

    limit = min(len(pdf_reader.pages), 30)
    for title, declared in candidates[:8]:
        if not isinstance(declared, int):
            continue
        needle = title.upper()[:60]
        if not needle:
            continue
        for i in range(limit):
            text = (pdf_reader.pages[i].extract_text() or "").upper()
            if needle in text:
                return (i + 1) - declared
    return 0

def correct_end_pages(toc, total_pages):
    def _int_start(item):
        sp = item.get("start_page")
        return sp if isinstance(sp, int) else 10**9

    for section in toc:
        if not isinstance(section.get("start_page"), int):
            subs = section.get("subsections", [])
            for sub in subs:
                ss = sub.get("start_page")
                if isinstance(ss, int):
                    section["start_page"] = ss
                    break

    toc.sort(key=_int_start)
    for section in toc:
        section.get("subsections", []).sort(key=_int_start)

    for i, section in enumerate(toc):
        next_start = None
        for j in range(i + 1, len(toc)):
            s2 = toc[j].get("start_page")
            if isinstance(s2, int):
                next_start = s2
                break
        if isinstance(next_start, int):
            section["end_page"] = max(1, min(total_pages, next_start - 1))
        else:
            section["end_page"] = total_pages

        subs = section.get("subsections", [])
        for j, sub in enumerate(subs):
            next_sub_start = None
            for k in range(j + 1, len(subs)):
                s2 = subs[k].get("start_page")
                if isinstance(s2, int):
                    next_sub_start = s2
                    break
            if isinstance(next_sub_start, int):
                sub["end_page"] = max(1, min(total_pages, next_sub_start - 1))
            else:
                sub["end_page"] = section["end_page"]

    for section in toc:
        s = section.get("start_page")
        e = section.get("end_page")
        if isinstance(s, int):
            section["start_page"] = max(1, min(total_pages, s))
        if isinstance(e, int):
            section["end_page"] = max(1, min(total_pages, e))
        if isinstance(section.get("start_page"), int) and isinstance(section.get("end_page"), int):
            if section["end_page"] < section["start_page"]:
                section["end_page"] = section["start_page"]
        for sub in section.get("subsections", []):
            ss = sub.get("start_page")
            ee = sub.get("end_page")
            if isinstance(ss, int):
                sub["start_page"] = max(1, min(total_pages, ss))
            if isinstance(ee, int):
                sub["end_page"] = max(1, min(total_pages, ee))
            if isinstance(sub.get("start_page"), int) and isinstance(sub.get("end_page"), int):
                if sub["end_page"] < sub["start_page"]:
                    sub["end_page"] = sub["start_page"]

    return toc

def extract_and_correct_toc(filepath, page_offset=0, max_toc_pages=5):
    pdf_reader = PdfReader(filepath)
    pass1_toc, pass1_toc_page_number = _extract_toc_internal(pdf_reader, page_offset=0, max_toc_pages=max_toc_pages)
    inferred = page_offset if isinstance(page_offset, int) and page_offset != 0 else infer_page_offset(pdf_reader, pass1_toc)
    toc, toc_page_number = _extract_toc_internal(pdf_reader, page_offset=inferred, max_toc_pages=max_toc_pages)

    total_pages = len(pdf_reader.pages)
    for section in toc:
        s = section.get("start_page"); e = section.get("end_page")
        if isinstance(s, int):
            section["start_page"] = max(1, min(total_pages, s))
        if isinstance(e, int):
            section["end_page"] = max(1, min(total_pages, e))
        if isinstance(section.get("start_page"), int) and isinstance(section.get("end_page"), int):
            if section["end_page"] < section["start_page"]:
                section["end_page"] = section["start_page"]
        for sub in section.get("subsections", []):
            ss = sub.get("start_page"); ee = sub.get("end_page")
            if isinstance(ss, int):
                sub["start_page"] = max(1, min(total_pages, ss))
            if isinstance(ee, int):
                sub["end_page"] = max(1, min(total_pages, ee))
            if isinstance(sub.get("start_page"), int) and isinstance(sub.get("end_page"), int):
                if sub["end_page"] < sub["start_page"]:
                    sub["end_page"] = sub["start_page"]

    toc = correct_end_pages(toc, total_pages)
    for section in toc:
        for sub in section["subsections"]:
            if not isinstance(sub.get("start_page"), int):
                sub["start_page"] = section.get("start_page")

    return toc, toc_page_number, pdf_reader

def _roman_from_section_title(title: str) -> str:
    m = re.search(r"SECTION\s+([IVXLCDM]+)", title, re.IGNORECASE)
    return m.group(1) if m else ""

def toc_to_dataframe(toc):
    data = []
    for i, section in enumerate(toc):
        roman = _roman_from_section_title(section["section"])
        for j, subsection in enumerate(section["subsections"]):
            start = subsection.get("start_page", None)
            end = subsection.get("end_page", None)
            tag_id = f"{i + 1}.{j + 1}"
            data.append({
                "TOC Tag ID": tag_id,
                "Section No. (Roman)": roman,
                "Section": section["section"],
                "Sub-section": subsection["subsection"],
                "Start Page#": start,
                "End Page#": end
            })
    return pd.DataFrame(data)