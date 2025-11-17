import os
import json
import logging
from datetime import datetime

from app import db
from models.dhrp_entry import DhrpEntry, ProcessingStatus, QaResult, RiskSummary, TocSection


def get_base_name(filename: str) -> str:
    """Return the base name of a file without extension."""
    return os.path.splitext(os.path.basename(filename))[0]

def normalize_name(name: str) -> str:
    """Normalize names to lowercase and replace spaces with underscores."""
    return name.lower().replace(" ", "_")



INDEX_FILE = os.path.join("dhrp_index.json")

def load_index() -> list:
    """Load index entries from JSON file."""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_index(entries: list) -> None:
    """Save index entries to JSON file."""
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2)



STATUS_PATH = os.path.join("processing_status.json")

def load_processing_status() -> dict:
    """Load processing status from JSON file, create if missing."""
    if not os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_processing_status(status_dict: dict) -> None:
    """Save processing status to JSON file."""
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)



def normalize_hierarchy_ranges(hierarchy: dict, total_pages: int | None = None) -> dict:
    """
    Clamp and fix section/subsection ranges so that:
      - 1 <= pdf_start_page/pdf_end_page <= total_pages (if provided)
      - pdf_end_page >= pdf_start_page
    """
    if not isinstance(hierarchy, dict):
        return hierarchy

    for _, data in hierarchy.items():
        s = data.get("pdf_start_page")
        e = data.get("pdf_end_page")
        if isinstance(s, int) and isinstance(e, int):
            if total_pages is not None:
                s = max(1, min(total_pages, s))
                e = max(1, min(total_pages, e))
            if e < s:
                e = s
            data["pdf_start_page"], data["pdf_end_page"] = s, e

        subs = data.get("subsections", {}) or {}
        for _, sub in subs.items():
            ss = sub.get("pdf_start_page")
            ee = sub.get("pdf_end_page")
            if isinstance(ss, int) and isinstance(ee, int):
                if total_pages is not None:
                    ss = max(1, min(total_pages, ss))
                    ee = max(1, min(total_pages, ee))
                if ee < ss:
                    ee = ss
                sub["pdf_start_page"], sub["pdf_end_page"] = ss, ee
    return hierarchy


def update_processing_stage(base: str, stage: str) -> None:
    """Update processing stage for a given DHRP entry."""
    if stage.startswith("🧩"):
        return

    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        logging.warning(f"⚠️ Cannot update stage — entry not found for {base}")
        return

    status = ProcessingStatus.query.filter_by(dhrp_id=entry.id).first()
    if not status:
        status = ProcessingStatus(dhrp_id=entry.id)

    MAX_LENGTH = 250
    status.processing_stage = stage[:MAX_LENGTH]
    status.updated_at = datetime.now().isoformat(timespec="seconds")

    db.session.add(status)
    db.session.commit()
    logging.info(f"🔄 Stage updated to '{stage[:MAX_LENGTH]}' for {base}")

def save_toc_to_db(base: str, toc_dict: dict) -> None:
    """Save TOC hierarchy to database."""
    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        logging.warning(f"⚠️ Cannot save TOC — entry not found for {base}")
        return

    for title, data in toc_dict.items():
        main_page = data.get("start_page", 0)
        db.session.add(TocSection(
            dhrp_id=entry.id,
            title=title,
            page=main_page
        ))
        for sub_title, sub_data in data.get("subsections", {}).items():
            db.session.add(TocSection(
                dhrp_id=entry.id,
                title=title,
                page=main_page,
                subsection_title=sub_title,
                subsection_page=sub_data.get("start_page", 0)
            ))
    db.session.commit()
    logging.info(f"✅ TOC saved to DB for {base}")

def save_qa_results_to_db(base: str, questions: list, answers: list) -> None:
    """Save Q&A results to database."""
    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        logging.warning(f"⚠️ Cannot save Q&A — entry not found for {base}")
        return

    for q, a in zip(questions, answers):
        db.session.add(QaResult(
            dhrp_id=entry.id,
            question=q,
            answer=a
        ))
    db.session.commit()
    logging.info(f"✅ Q&A results saved to DB for {base}")

def save_risk_summary_to_db(base: str, risk_text: str, summary_bullets: list) -> None:
    """Save risk summary to database."""
    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        logging.warning(f"⚠️ Cannot save risk summary — entry not found for {base}")
        return

    summary = RiskSummary.query.filter_by(dhrp_id=entry.id).first()
    if not summary:
        summary = RiskSummary(dhrp_id=entry.id)

    summary.risk_text = risk_text
    summary.summary_bullets = json.dumps(summary_bullets)

    db.session.add(summary)
    db.session.commit()
    logging.info(f"✅ Risk summary saved to DB for {base}")



os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs", "activity.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
