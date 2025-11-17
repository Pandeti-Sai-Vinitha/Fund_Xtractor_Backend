import os

import json
import pickle
import fitz
import requests
import numpy as np
import pandas as pd
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from PyPDF2 import PdfReader

from detect_dhrp import is_dhrp

from section import extract_toc_sections_hierarchical
from services.chunker import chunk_pdf_by_toc
from services.core_logic import build_llm_prompt_for_fund_setup, evaluate_with_gemini, get_chunks_with_fallback, parse_gemini_response
from services.embedder import embed_chunks_optimal
from services.risk_summary import generate_risk_summary_from_chunks
try:
    # Prefer importing the logic-only TOC utilities if available
    from section1 import extract_and_correct_toc, toc_to_dataframe, extract_text_range  # type: ignore
except Exception:
    extract_and_correct_toc = None  # type: ignore
    toc_to_dataframe = None  # type: ignore
    extract_text_range = None  # type: ignore
from eligibility_evaluator import process_csv_and_evaluate

import threading
import config
from models.models import db, DhrpEntry, ProcessingStatus, TocSection, RiskSummary, QaResult

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
app.config.from_object(config)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:4200",
    "*"
]}}, supports_credentials=True)

db.init_app(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('pickles', exist_ok=True)
os.makedirs('toc', exist_ok=True)
os.makedirs('risk_summary', exist_ok=True)
os.makedirs('answered_csv', exist_ok=True)
os.makedirs('questions_csv', exist_ok=True)
os.makedirs('cache', exist_ok=True)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_FILE = os.path.join(BASE_DIR, 'dhrp_index.json')
API_TOC_DIR = os.path.join(BASE_DIR, 'toc')

os.makedirs(API_TOC_DIR, exist_ok=True)

def get_base_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def normalize_name(name):
    return name.lower().replace(" ", "_")

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_index(entries):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2)

STATUS_PATH = os.path.join("processing_status.json")

def load_processing_status():
    if not os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_processing_status(status_dict):
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

def update_processing_stage(base, stage):
    # Skip verbose Gemini reasoning steps from DB logging
    if stage.startswith("🧩"):
        return

    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        logging.warning(f"⚠️ Cannot update stage — entry not found for {base}")
        return

    status = ProcessingStatus.query.filter_by(dhrp_id=entry.id).first()
    if not status:
        status = ProcessingStatus(dhrp_id=entry.id)

    # Truncate stage to avoid overflow
    MAX_LENGTH = 250
    status.processing_stage = stage[:MAX_LENGTH]
    status.updated_at = datetime.now().isoformat(timespec="seconds")

    db.session.add(status)
    db.session.commit()
    logging.info(f"🔄 Stage updated to '{stage[:MAX_LENGTH]}' for {base}")


def save_toc_to_db(base, toc_dict):
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

def save_qa_results_to_db(base, questions, answers):
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

def save_risk_summary_to_db(base, risk_text, summary_bullets):
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

import logging
from datetime import datetime

# Create logs folder if needed
os.makedirs("logs", exist_ok=True)

# Configure logger
logging.basicConfig(
    filename=os.path.join("logs", "activity.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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


@app.route('/upload', methods=['POST'])
def upload_dhrp():
    try:
        company = request.form['company']
        bse_code = request.form['bse_code']
        upload_date = request.form['upload_date']
        uploader_name = request.form['uploader_name']
        promoter = request.form['promoter']
        pdf = request.files['pdf']
        if not pdf or pdf.filename == '':
            logging.warning(f"Upload failed — No PDF provided by {uploader_name} for {company}")
            return jsonify({"success": False, "message": "No PDF uploaded"}), 400

        original_filename = secure_filename(pdf.filename)
        base = normalize_name(get_base_name(original_filename))
        filename = f"{base}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf.save(pdf_path)
        logging.info(f"📄 PDF saved: {filename} by {uploader_name} for {company}")

        entry = DhrpEntry(
            company=company,
            bse_code=bse_code,
            upload_date=upload_date,
            uploader_name=uploader_name,
            promoter=promoter,
            pdf_filename=filename,
            status="Processing"
        )
        db.session.add(entry)
        db.session.commit()
        logging.info(f"📥 DHRP entry saved to database: {filename}")

        index_path = os.path.join(BASE_DIR, 'dhrp_index.json')

        new_index_entry = {
            "company": company.strip(),
            "bse_code": bse_code.strip(),
            "upload_date": upload_date.strip(),
            "uploader_name": uploader_name.strip(),
            "promoter": promoter.strip(),
            "pdf_filename": filename.strip().lower(),
            "status": "New",
            "toc_verified": False
        }
        try:
            with open(index_path, 'r+', encoding='utf-8') as f:
                entries = json.load(f)
                entries.append(new_index_entry)
                f.seek(0)
                json.dump(entries, f, indent=2)
                f.truncate()
            logging.info(f"🗂️ Entry added to index: {filename}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to update index for {filename}: {str(e)}")

        # ======================
        # TOC extraction (new)
        # ======================
        toc_path = os.path.join('toc', f"{base}.json")
        os.makedirs('toc', exist_ok=True)

        if extract_and_correct_toc is None:
            return jsonify({"success": False, "message": "TOC extractor not available"}), 500

        toc_list, toc_page_number, _ = extract_and_correct_toc(pdf_path, page_offset=0)

        hierarchy = {}
        for section in toc_list:
            title = str(section.get("section", "")).strip()
            s_start = section.get("start_page")
            s_end = section.get("end_page")

            if title not in hierarchy:
                hierarchy[title] = {
                    "start_page": str(s_start) if isinstance(s_start, int) else None,
                    "end_page": str(s_end) if isinstance(s_end, int) else None,
                    "pdf_start_page": s_start if isinstance(s_start, int) else None,
                    "pdf_end_page": s_end if isinstance(s_end, int) else None,
                    "subsections": {},
                }

            sec = hierarchy[title]
            if isinstance(s_start, int):
                if sec["pdf_start_page"] is None or s_start < sec["pdf_start_page"]:
                    sec["pdf_start_page"] = s_start
                    sec["start_page"] = str(s_start)
            if isinstance(s_end, int):
                if sec["pdf_end_page"] is None or s_end > sec["pdf_end_page"]:
                    sec["pdf_end_page"] = s_end
                    sec["end_page"] = str(s_end)

            for sub in section.get("subsections", []):
                sub_title = str(sub.get("subsection", "")).strip()
                sub_start = sub.get("start_page")
                sub_end = sub.get("end_page")
                if not sub_title:
                    continue
                sec["subsections"][sub_title] = {
                    "start_page": str(sub_start) if isinstance(sub_start, int) else None,
                    "end_page": str(sub_end) if isinstance(sub_end, int) else None,
                    "pdf_start_page": sub_start if isinstance(sub_start, int) else None,
                    "pdf_end_page": sub_end if isinstance(sub_end, int) else None,
                }
                if isinstance(sub_start, int):
                    if sec["pdf_start_page"] is None or sub_start < sec["pdf_start_page"]:
                        sec["pdf_start_page"] = sub_start
                        sec["start_page"] = str(sub_start)
                if isinstance(sub_end, int):
                    if sec["pdf_end_page"] is None or sub_end > sec["pdf_end_page"]:
                        sec["pdf_end_page"] = sub_end
                        sec["end_page"] = str(sub_end)

        total_pages = len(PdfReader(pdf_path).pages)
        hierarchy = normalize_hierarchy_ranges(hierarchy, total_pages=total_pages)

        with open(toc_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2)
        logging.info(f"📑 TOC extracted and saved to file: {toc_path}")

        if isinstance(hierarchy, dict):
            save_toc_to_db(base, hierarchy)
        else:
            logging.warning(f"⚠️ TOC extraction returned unexpected format for {base}")

        return jsonify({
            "success": True,
            "message": "DHRP uploaded successfully",
            "entry": {
                "company": company,
                "bse_code": bse_code,
                "upload_date": upload_date,
                "uploader_name": uploader_name,
                "promoter": promoter,
                "pdf_filename": filename,
                "status": "New"
            },
            "base": base
        }), 200

    except Exception as e:
        logging.error(f"❌ Upload error for {request.form.get('company', 'Unknown')}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/verify_toc/<base>', methods=['POST'])
def mark_toc_verified(base):
    try:
        filename = f"{base}.pdf"
        entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
        if not entry:
            logging.warning(f"⚠️ TOC verification failed — Entry not found for: {filename}")
            return jsonify({"success": False, "message": "Entry not found"}), 404
        entry.toc_verified = True
        db.session.commit()
        logging.info(f"✅ TOC verified for: {filename} — Company: {entry.company}")
        return jsonify({"success": True, "message": "TOC marked as verified"}), 200
    except Exception as e:
        logging.error(f"❌ Error verifying TOC for {base}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/accept_toc/<base>', methods=['POST'])
def accept_toc(base):
    try:
        filename = f"{base}.pdf"
        entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
        if not entry:
            logging.warning(f"⚠️ TOC acceptance failed — Entry not found for: {filename}")
            return jsonify({"success": False, "message": "Entry not found"}), 404
        entry.toc_verified = True
        db.session.commit()
        logging.info(f"✅ TOC accepted for: {filename} — Company: {entry.company}")
        return jsonify({"success": True, "message": "TOC accepted and verified"}), 200
    except Exception as e:
        logging.error(f"❌ Error accepting TOC for {base}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/save_toc/<doc>')
def save_toc_from_csv(doc):
    base_name = os.path.splitext(os.path.basename(doc))[0]
    csv_name = f"{base_name}.csv"
    json_name = f"{base_name}.json"
    csv_path = os.path.join(API_TOC_DIR, csv_name)
    json_path = os.path.join(API_TOC_DIR, json_name)

    # Fallback: synthesize CSV from canonical toc/<base>.json if missing
    if not os.path.exists(csv_path):
        canonical_toc_path = os.path.join(API_TOC_DIR, f"{base_name}.json")
        if not os.path.exists(canonical_toc_path):
            return jsonify({"error": f"CSV not found for {doc} and no canonical TOC at {canonical_toc_path}"}), 404

        with open(canonical_toc_path, "r", encoding="utf-8") as f:
            canonical = json.load(f)

        rows = []
        tag_i = 1
        for section_title, sec in canonical.items():
            subs = sec.get("subsections", {}) or {}
            if isinstance(subs, dict) and subs:
                tag_j = 1
                for sub_title, sub in subs.items():
                    rows.append({
                        "TOC Tag ID": f"{tag_i}.{tag_j}",
                        "Section No. (Roman)": "",
                        "Section": section_title,
                        "Sub-section": sub_title,
                        "Start Page#": sub.get("pdf_start_page"),
                        "End Page#": sub.get("pdf_end_page"),
                    })
                    tag_j += 1
            else:
                rows.append({
                    "TOC Tag ID": f"{tag_i}.1",
                    "Section No. (Roman)": "",
                    "Section": section_title,
                    "Sub-section": section_title,
                    "Start Page#": sec.get("pdf_start_page"),
                    "End Page#": sec.get("pdf_end_page"),
                })
            tag_i += 1

        pd.DataFrame(rows, columns=[
            "TOC Tag ID",
            "Section No. (Roman)",
            "Section",
            "Sub-section",
            "Start Page#",
            "End Page#",
        ]).to_csv(csv_path, index=False)

    # Read CSV and build hierarchy
    df = pd.read_csv(csv_path)
    hierarchy: dict = {}

    for _, row in df.iterrows():
        section = str(row.get("Section", "")).strip()
        subsection = str(row.get("Sub-section", "")).strip()
        start_val = row.get("Start Page#")
        end_val = row.get("End Page#")
        start_int = int(start_val) if pd.notna(start_val) else None
        end_int = int(end_val) if pd.notna(end_val) else None

        # ✅ Fix reversed page ranges
        if isinstance(start_int, int) and isinstance(end_int, int) and start_int > end_int:
            logging.warning(f"⚠️ Reversed range in {section} → {subsection or 'FULL'}: {start_int} > {end_int}")
            end_int = start_int

        if section not in hierarchy:
            hierarchy[section] = {
                "start_page": str(start_int) if isinstance(start_int, int) else None,
                "end_page": str(end_int) if isinstance(end_int, int) else None,
                "pdf_start_page": start_int,
                "pdf_end_page": end_int,
                "subsections": {},
            }

        sec = hierarchy[section]
        if isinstance(start_int, int):
            if sec["pdf_start_page"] is None or start_int < sec["pdf_start_page"]:
                sec["pdf_start_page"] = start_int
                sec["start_page"] = str(start_int)
        if isinstance(end_int, int):
            if sec["pdf_end_page"] is None or end_int > sec["pdf_end_page"]:
                sec["pdf_end_page"] = end_int
                sec["end_page"] = str(end_int)

        if subsection:
            sec["subsections"][subsection] = {
                "start_page": str(start_int) if isinstance(start_int, int) else None,
                "end_page": str(end_int) if isinstance(end_int, int) else None,
                "pdf_start_page": start_int,
                "pdf_end_page": end_int,
            }

    # Normalize using actual PDF length if present
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}.pdf")
    total_pages = len(PdfReader(pdf_path).pages) if os.path.exists(pdf_path) else None
    hierarchy = normalize_hierarchy_ranges(hierarchy, total_pages=total_pages)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"hierarchy": hierarchy}, jf, indent=2)

    return jsonify({
        "csvPath": csv_path,
        "jsonPath": json_path,
        "hierarchy": hierarchy
    })


@app.route('/api/save_toc', methods=['POST'])
def save_edited_toc():
    try:
        data = request.get_json()
        pdf_name = data.get("pdfName")
        rows = data.get("rows", [])
        if not pdf_name or not rows:
            return jsonify({"error": "Missing PDF name or TOC rows"}), 400

        base_name = os.path.splitext(os.path.basename(pdf_name))[0]
        csv_path = os.path.join(API_TOC_DIR, f"{base_name}.csv")
        json_path = os.path.join(API_TOC_DIR, f"{base_name}.json")

        # Save edited rows to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Reuse your existing logic to rebuild hierarchy from CSV
        return save_toc_from_csv(pdf_name)

    except Exception as e:
        logging.error(f"❌ Failed to save edited TOC: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/replace_toc_file', methods=['POST'])
def replace_toc_file():
    uploaded_file = request.files.get('file')
    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    
    filename = secure_filename(uploaded_file.filename)
    base_name = normalize_name(get_base_name(filename))
    os.makedirs('toc', exist_ok=True)
    
    if filename.endswith('.json'):
        json_path = os.path.join('toc', f"{base_name}.json")
        uploaded_file.save(json_path)
        return jsonify({
            "success": True,
            "message": f"TOC JSON replaced for {base_name}",
            "path": json_path
        })
    elif filename.endswith('.csv'):
        csv_path = os.path.join('toc', f"{base_name}_edited_toc.csv")
        uploaded_file.save(csv_path)
        return jsonify({
            "success": True,
            "message": f"TOC CSV replaced for {base_name}",
            "path": csv_path
        })
    else:
        return jsonify({"success": False, "message": "Only .json or .csv files are accepted"}), 400

@app.route('/reject_toc/<base>', methods=['POST'])
def reject_toc(base):
    try:
        toc_path = os.path.join('toc', f"{base}.json")
        if os.path.exists(toc_path):
            os.remove(toc_path)
            logging.info(f"🗑️ TOC file deleted: {toc_path}")
        else:
            logging.warning(f"⚠️ TOC file not found for deletion: {toc_path}")

        filename = f"{base}.pdf"
        entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
        if entry:
            TocSection.query.filter_by(dhrp_id=entry.id).delete()
            RiskSummary.query.filter_by(dhrp_id=entry.id).delete()
            ProcessingStatus.query.filter_by(dhrp_id=entry.id).delete()
            db.session.delete(entry)
            db.session.commit()
            logging.info(f"🗂️ DHRP entry and related records removed for: {filename} — Company: {entry.company}")
        else:
            logging.warning(f"⚠️ No matching DB entry found for base: {base}")

        return jsonify({"success": True, "message": f"TOC and index entry for '{base}' removed"}), 200
    except Exception as e:
        logging.error(f"❌ Error rejecting TOC for {base}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/process/<base>', methods=['POST'])
def process_dhrp(base):
    try:
        filename = f"{base}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(pdf_path):
            logging.warning(f"⚠️ PDF not found: {pdf_path}")
            return jsonify({"success": False, "message": f"PDF not found for {base}"}), 404

        entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
        if not entry:
            logging.warning(f"⚠️ Entry not found for: {filename}")
            return jsonify({"success": False, "message": "Entry not found"}), 404

        logging.info(f"🚀 Starting background processing for: {filename} — Company: {entry.company}")

        def run_background_process(base, entry):
            with app.app_context():
                background_process_dhrp(base, entry)

        thread = threading.Thread(target=run_background_process, args=(base, entry))
        thread.start()

        return jsonify({
            "success": True,
            "message": f"Processing started for {entry.company}. You can continue using the dashboard.",
            "base": base
        }), 202

    except Exception as e:
        logging.error(f"❌ Error initiating processing for {base}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


def background_process_dhrp(base, entry):
    try:
        filename = f"{base}.pdf"
        company = entry.company or "Unknown"
        source_pdf_filename = entry.pdf_filename.strip().lower()
        logging.info(f"📄 Background task started for: {filename} — Company: {company}")

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        toc_path = os.path.join('toc', f"{base}.json")
        risk_path = os.path.join('risk_summary', f"{base}.json")
        raw_pkl_path = os.path.join('pickles', f"{base}.pkl")
        embedded_pkl_path = os.path.join('pickles', f"{base}_embedded.pkl")
        csv_path = os.path.join('questions_csv', 'Fund Set-up dow jones.csv')
        answered_csv_path = os.path.join('answered_csv', f"{base}_analysis.csv")
        index_path = os.path.join(BASE_DIR, 'dhrp_index.json')

        for label, path in [("PDF", pdf_path), ("TOC", toc_path)]:
            if not path or not isinstance(path, str):
                raise ValueError(f"{label} path is invalid: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        if not os.path.exists(csv_path):
            logging.warning(f"⚠️ CSV template missing — creating fallback at: {csv_path}")
            fallback_questions = [
                "What is the business model?",
                "What are the key financial risks?",
                "Does the company meet profitability criteria?"
            ]
            pd.DataFrame({"Question": fallback_questions}).to_csv(csv_path, index=False)

        update_processing_stage(base, "🔧 Chunking PDF by TOC")
        chunk_pdf_by_toc(pdf_path, toc_path, raw_pkl_path)

        update_processing_stage(base, "🔗 Embedding chunks")
        embed_chunks_optimal(raw_pkl_path, embedded_pkl_path)

        update_processing_stage(base, "📦 Loading embedded chunks")
        if not os.path.exists(embedded_pkl_path):
            raise FileNotFoundError(f"Embedded chunks not found: {embedded_pkl_path}")
        with open(embedded_pkl_path, 'rb') as f:
            embedded_chunks = pickle.load(f)

        update_processing_stage(base, "🧠 Generating risk summary")
        risk_text, summary_bullets, top_chunks = generate_risk_summary_from_chunks(
            chunks=embedded_chunks,
            user_question="What are the key risk factors disclosed in the DHRP?",
            search_intent="Search 'Risk Factors', 'Business Overview', and 'Management Discussion and Analysis' for risk-related disclosures.",
            remarks="Focus on financial, operational, regulatory, and competitive risks. Include any mitigation strategies or gaps.",
            cache_dir="cache"
        )

        update_processing_stage(base, "📝 Saving risk summary")
        with open(risk_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary_bullets": summary_bullets,
                "risk_text": risk_text
            }, f, indent=2)
        save_risk_summary_to_db(base, risk_text, summary_bullets)

        update_processing_stage(base, "📊 Evaluating Q&A CSV")
        with open(index_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        matched_entry = next(
            (e for e in entries if e.get("pdf_filename", "").strip().lower() == source_pdf_filename),
            None
        )
        if not matched_entry:
            raise ValueError(f"No entry found in index for PDF: {source_pdf_filename}")

        stream_csv_evaluation(
            base=base,
            chunks=embedded_chunks,
            csv_path=csv_path,
            output_path=answered_csv_path
        )

        update_processing_stage(base, "📈 Finalizing status")
        for e in entries:
            if e.get("pdf_filename", "").strip().lower() == source_pdf_filename:
                e["status"] = "Completed"
                break
        save_index(entries)

        db_entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
        if db_entry:
            db_entry.status = "Completed"
            db.session.commit()
        else:
            logging.warning(f"⚠️ Could not update status — entry not found for: {filename}")

        update_processing_stage(base, "✅ Completed")
        logging.info(f"📈 Status updated to 'Completed' for: {filename}")

    except Exception as e:
        update_processing_stage(base, "❌ Error during processing")
        logging.error(f"❌ Error during background processing for {base}: {str(e)}")

@app.route('/stream-status/<base>')
def get_status(base):
    status_path = os.path.join('status', f"{base}.json")

    if not os.path.exists(status_path):
        # Explicitly set content type to JSON and avoid default HTML error page
        response = jsonify({"success": False, "message": "Status not available yet."})
        response.status_code = 200  # or 202 if you want to indicate "still processing"
        return response

    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        return jsonify({"success": True, "base": base, "status": status_data})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading status file: {str(e)}"}), 500



@app.route('/status/<base>', methods=['GET'])
def get_processing_status(base):
    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        return jsonify({"success": False, "message": "Entry not found"}), 404
    status = ProcessingStatus.query.filter_by(dhrp_id=entry.id).first()
    if not status:
        return jsonify({"success": False, "message": "No status found"}), 404
    return jsonify({
        "success": True,
        "stage": status.processing_stage,
        "updated_at": status.updated_at
    })

@app.route('/get_all_dhrps')
def get_all_dhrps():
    try:
        entries = DhrpEntry.query.all()
        result = [{
            "company": e.company,
            "bse_code": e.bse_code,
            "upload_date": e.upload_date,
            "uploader_name": e.uploader_name,
            "promoter": e.promoter,
            "pdf_filename": e.pdf_filename,
            "status": e.status,
            "toc_verified": e.toc_verified
        } for e in entries]
        logging.info(f"📥 DHRP entries retrieved — {len(result)} entries")
        return jsonify(result)
    except Exception as e:
        logging.error(f"❌ Error loading DHRP entries: {str(e)}")
        return jsonify([]), 500

@app.route('/get_toc/<doc>')
def get_toc(doc):
    try:
        base = normalize_name(get_base_name(doc))
        toc_path = os.path.join('toc', f"{base}.json")
        
        # First, try to read from uploaded JSON file
        if os.path.exists(toc_path):
            try:
                with open(toc_path, 'r', encoding='utf-8') as f:
                    toc_data = json.load(f)
                
                # Handle JSON wrapped in "hierarchy" key
                if isinstance(toc_data, dict) and "hierarchy" in toc_data:
                    toc_dict = toc_data["hierarchy"]
                else:
                    toc_dict = toc_data
                
                if toc_dict:
                    toc_list = []
                    for title, data in toc_dict.items():
                        # Extract page number - prefer pdf_start_page, then start_page, then page
                        page_num = data.get("pdf_start_page") or data.get("start_page") or data.get("page")
                        if isinstance(page_num, str):
                            try:
                                page_num = int(page_num)
                            except (ValueError, TypeError):
                                page_num = None
                        
                        # Handle subsections
                        subs = data.get("subsections", {})
                        subs_list = []
                        if isinstance(subs, dict):
                            for sub_title, sub_data in subs.items():
                                # Extract subsection page - prefer pdf_start_page, then start_page
                                sub_page = sub_data.get("pdf_start_page") if isinstance(sub_data, dict) else None
                                if sub_page is None and isinstance(sub_data, dict):
                                    sub_page = sub_data.get("start_page")
                                if sub_page is None:
                                    sub_page = sub_data if isinstance(sub_data, (int, str)) else None
                                if isinstance(sub_page, str):
                                    try:
                                        sub_page = int(sub_page)
                                    except (ValueError, TypeError):
                                        sub_page = None
                                subs_list.append({
                                    "title": sub_title,
                                    "page": sub_page
                                })
                        elif isinstance(subs, list):
                            subs_list = subs
                        else:
                            subs_list = []
                        
                        toc_list.append({
                            "title": title,
                            "page": page_num,
                            "subsections": subs_list
                        })
                    
                    if toc_list:
                        return jsonify({"toc": toc_list})
            except Exception as e:
                # If JSON read fails, fall through to database
                pass
        
        # Fallback to database if JSON file doesn't exist or is empty
        try:
            from models import DhrpEntry, TocSection
            entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
            if not entry:
                return jsonify({"toc": []})
            
            sections = TocSection.query.filter_by(dhrp_id=entry.id).all()
            if not sections:
                return jsonify({"toc": []})
            
            toc_dict = {}
            for section in sections:
                title = section.title
                if title not in toc_dict:
                    toc_dict[title] = {
                        "page": section.page,
                        "subsections": []
                    }
                if section.subsection_title:
                    toc_dict[title]["subsections"].append({
                        "title": section.subsection_title,
                        "page": section.subsection_page
                    })
            
            toc_list = [
                {
                    "title": title,
                    "page": data["page"],
                    "subsections": data["subsections"]
                }
                for title, data in toc_dict.items()
            ]
            
            return jsonify({"toc": toc_list})
        except ImportError:
            # Database models not available, return empty
            return jsonify({"toc": []})
        except Exception as e:
            return jsonify({"toc": []}), 500
            
    except Exception as e:
        return jsonify({"toc": []}), 500

@app.route('/risk/<doc>')
def get_risk(doc):
    try:
        base = normalize_name(get_base_name(doc))
        entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
        if not entry:
            logging.warning(f"⚠️ DHRP entry not found for: {doc}")
            return jsonify({"success": False, "message": "DHRP entry not found"}), 404
        summary = RiskSummary.query.filter_by(dhrp_id=entry.id).first()
        if not summary:
            logging.warning(f"⚠️ Risk summary not found in DB for: {doc}")
            return jsonify({"success": False, "message": "Risk summary not found"}), 404
        risk_text = summary.risk_text or ""
        summary_bullets = json.loads(summary.summary_bullets or "{}")
        logging.info(f"📊 Risk summary served from DB for: {doc} — Bullets: {len(summary_bullets)}")
        return jsonify({
            "success": True,
            "doc": doc,
            "risk_text": risk_text,
            "summary_bullets": summary_bullets
        })
    except Exception as e:
        logging.error(f"❌ Error loading risk summary from DB for {doc}: {str(e)}")
        return jsonify({"success": False, "message": f"Failed to load summary: {str(e)}"}), 500

@app.route('/delete/<doc>', methods=['POST'])
def delete_doc(doc):
    try:
        base = normalize_name(get_base_name(doc))
        filename = f"{base}.pdf"
        paths = [
            f"uploads/{base}.pdf",
            f"pickles/{base}.pkl",
            f"pickles/{base}_embedded.pkl",
            f"toc/{base}.json",
            f"risk_summary/{base}.json",
            f"answered_csv/{base}_analysis.csv"
        ]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"🗑️ File deleted: {path}")
            else:
                logging.warning(f"⚠️ File not found for deletion: {path}")

        entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
        if entry:
            TocSection.query.filter_by(dhrp_id=entry.id).delete()
            RiskSummary.query.filter_by(dhrp_id=entry.id).delete()
            ProcessingStatus.query.filter_by(dhrp_id=entry.id).delete()
            db.session.delete(entry)
            db.session.commit()
            logging.info(f"🗂️ DB entry and related records removed for: {filename} — Company: {entry.company}")
        else:
            logging.warning(f"⚠️ No matching DB entry found for base: {base}")

        return jsonify({"message": f"{doc} deleted successfully"})
    except Exception as e:
        logging.error(f"❌ Error deleting document {doc}: {str(e)}")
        return jsonify({"message": f"Error deleting {doc}: {str(e)}"}), 500

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

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    try:
        selected_doc = request.form.get('selected_doc')
        csv_file = request.files.get('csv')
        if not selected_doc or not csv_file or csv_file.filename == '':
            logging.warning("⚠️ CSV analysis failed — Missing document or CSV file")
            return jsonify({"error": "Missing document or CSV file"}), 400

        base = normalize_name(get_base_name(selected_doc))
        pkl_path = f"pickles/{base}.pkl"
        if not os.path.exists(pkl_path):
            logging.warning(f"⚠️ Pickle file not found for: {selected_doc}")
            return jsonify({"error": f"Pickle file not found for {selected_doc}"}), 404

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
        return jsonify({"message": "CSV analyzed successfully", "filename": f"{base}_analysis.csv"})
    except Exception as e:
        logging.error(f"❌ Error during CSV analysis for {selected_doc}: {str(e)}")
        return jsonify({"error": f"Failed to analyze CSV: {str(e)}"}), 500

@app.route('/view_csv/<filename>')
def view_csv(filename):
    try:
        csv_path = os.path.join('answered_csv', filename)
        if not os.path.exists(csv_path):
            logging.warning(f"⚠️ CSV file not found for viewing: {filename}")
            return jsonify({
                "success": False,
                "message": f"❌ CSV file not found: {filename}"
            }), 404

        df = pd.read_csv(csv_path)
        df = df.replace({np.nan: None})
        records = df.to_dict(orient='records')
        columns = list(df.columns)

        # Extract base name (without .csv)
        base_name = os.path.splitext(filename)[0]

        logging.info(f"📄 CSV viewed: {filename} — Rows: {len(records)}")
        return jsonify({
            "success": True,
            "filename": filename,
            "base_name": base_name,
            "columns": columns,
            "rows": records
        })
    except Exception as e:
        logging.error(f"❌ Error reading CSV {filename}: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"❌ Failed to read CSV: {str(e)}"
        }), 500


@app.route('/api/comments', methods=['GET'])
def get_comments():
    filename = request.args.get('filename')
    item_id = request.args.get('itemId', type=int)
    comments_path = "comments.json"
    if not os.path.exists(comments_path):
        return jsonify({"comments": []})
    with open(comments_path, "r") as f:
        all_comments = json.load(f)
    filtered = [c for c in all_comments if c.get("filename") == filename and c.get("itemId") == item_id]
    return jsonify({"comments": filtered})

@app.route('/api/comments', methods=['POST'])
def add_comment():
    comment = request.json
    comment["timestamp"] = datetime.now().isoformat()
    comments_path = "comments.json"
    comments = []

    logging.info(f"📝 Received comment: {comment}")

    # Load existing comments
    if os.path.exists(comments_path):
        with open(comments_path, "r") as f:
            comments = json.load(f)
        logging.info(f"📂 Loaded {len(comments)} existing comments")
    else:
        logging.info("📂 No existing comments file found — creating new")

    # Assign ID and save to comments.json
    comment["id"] = len(comments) + 1
    comments.append(comment)
    with open(comments_path, "w") as f:
        json.dump(comments, f, indent=2)
    logging.info(f"✅ Comment saved to comments.json with ID {comment['id']}")

    # Append comment to corresponding answer CSV row
    try:
        filename = comment.get("filename")  # e.g., "abc.pdf" or "abc_analysis.csv"
        item_id = comment.get("itemId")     # row index
        reviewer = comment.get("reviewer", "Unknown")
        base = normalize_name(get_base_name(filename))
        csv_path = os.path.join("answered_csv", f"{base}.csv")

        logging.info(f"📄 Target CSV: {csv_path} — Row index: {item_id}")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logging.info(f"📊 Loaded CSV with {len(df)} rows")

            # Ensure "Comment" column exists
            if "Comment" not in df.columns:
                df["Comment"] = ""
                logging.info("🧩 'Comment' column not found — added new column")

            # Append comment to the correct row
            if 0 <= item_id < len(df):
                existing = df.at[item_id, "Comment"]
                existing = "" if pd.isna(existing) else str(existing).strip()
                new_line = f"[{comment['timestamp']}] {reviewer}: {comment['comment']}"
                new_comment = f"{existing}\n{new_line}".strip() if existing else new_line
                df.at[item_id, "Comment"] = new_comment
                df.to_csv(csv_path, index=False)
                logging.info(f"✅ Comment appended to row {item_id} in CSV")
            else:
                logging.warning(f"⚠️ itemId {item_id} is out of bounds for CSV with {len(df)} rows")
        else:
            logging.warning(f"⚠️ CSV file not found: {csv_path}")
    except Exception as e:
        logging.error(f"❌ Failed to append comment to CSV: {str(e)}")

    return jsonify({"success": True, "comment": comment})





@app.route('/download/<filename>')
def download_csv(filename):
    try:
        path = os.path.join('answered_csv', filename)
        if not os.path.exists(path):
            logging.warning(f"⚠️ Download failed — File not found: {filename}")
            return jsonify({"error": "File not found"}), 404
        logging.info(f"📥 File download initiated: {filename}")
        return send_file(path, as_attachment=True)
    except Exception as e:
        logging.error(f"❌ Error during file download {filename}: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route('/get_company/<doc>')
def get_company(doc):
    try:
        entries = load_index()
        for entry in entries:
            if entry.get('pdf_filename') == doc:
                logging.info(f"🏢 Company details retrieved for: {doc} — Company: {entry.get('company', 'Unknown')}")
                return jsonify(entry)
        logging.warning(f"⚠️ Company details not found for: {doc}")
        return jsonify({"error": "Company details not found"}), 404
    except Exception as e:
        logging.error(f"❌ Error retrieving company details for {doc}: {str(e)}")
        return jsonify({"error": f"Failed to retrieve company details: {str(e)}"}), 500

@app.route('/get_traceability/<company>')
def get_traceability(company):
    trace_path = 'traceability.json'
    if not os.path.exists(trace_path):
        return jsonify({"history": []})
    with open(trace_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data.get(company, {"history": []}))

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            logging.warning("⚠️ CSV upload failed — No file part in request")
            return jsonify({'success': False, 'message': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            logging.warning("⚠️ CSV upload failed — No selected file")
            return jsonify({'success': False, 'message': 'No selected file'}), 400
        if not file.filename.endswith('.csv'):
            logging.warning(f"⚠️ CSV upload rejected — Invalid file type: {file.filename}")
            return jsonify({'success': False, 'message': 'Only CSV files are allowed'}), 400
        folder_path = os.path.join(os.getcwd(), 'questions_csv')
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, 'questions.csv')
        file.save(save_path)
        logging.info(f"📤 CSV uploaded successfully — Saved as: {save_path}")
        return jsonify({'success': True, 'message': 'CSV uploaded and replaced successfully'})
    except Exception as e:
        logging.error(f"❌ Error during CSV upload: {str(e)}")
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)