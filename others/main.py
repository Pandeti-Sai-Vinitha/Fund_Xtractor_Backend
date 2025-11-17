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
from chunker import chunk_pdf_by_toc
from embedder import embed_chunks_optimal
from section import extract_toc_sections_hierarchical
try:
    # Prefer importing the logic-only TOC utilities if available
    from section1 import extract_and_correct_toc, toc_to_dataframe, extract_text_range  # type: ignore
except Exception:
    extract_and_correct_toc = None  # type: ignore
    toc_to_dataframe = None  # type: ignore
    extract_text_range = None  # type: ignore
from eligibility_evaluator import process_csv_and_evaluate
from risk_summary import generate_risk_summary_from_chunks, parse_structured_risk_summary
import threading
import config
from models import db, DhrpEntry, ProcessingStatus, TocSection, RiskSummary, QaResult
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)

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

    
def update_processing_stage(base, stage):
    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        logging.warning(f"⚠️ Cannot update stage — entry not found for {base}")
        return
    status = ProcessingStatus.query.filter_by(dhrp_id=entry.id).first()
    if not status:
        status = ProcessingStatus(dhrp_id=entry.id)
    status.processing_stage = stage
    status.updated_at = datetime.now().isoformat(timespec="seconds")
    db.session.add(status)
    db.session.commit()
    logging.info(f"🔄 Stage updated to '{stage}' for {base}")


# Configure logger
logging.basicConfig(
    filename=os.path.join("logs", "activity.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
@app.route('/demo_upload', methods=['POST'])
def demo_upload_and_process():
    try:
        # Step 1: Extract metadata and file from request
        company = request.form['company']
        bse_code = request.form['bse_code']
        upload_date = request.form['upload_date']
        uploader_name = request.form['uploader_name']
        promoter = request.form['promoter']
        pdf = request.files['pdf']

        if not pdf or pdf.filename == '':
            logging.warning(f"Upload failed — No PDF provided by {uploader_name} for {company}")
            return jsonify({"success": False, "message": "No PDF uploaded"}), 400

        # Step 2: Normalize filename and save PDF
        original_filename = secure_filename(pdf.filename)
        base = normalize_name(get_base_name(original_filename))
        filename = f"{base}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf.save(pdf_path)
        logging.info(f"📄 PDF saved: {filename} by {uploader_name} for {company}")

        # Step 3: Create DB entry
        entry = DhrpEntry(
            company=company,
            bse_code=bse_code,
            upload_date=upload_date,
            uploader_name=uploader_name,
            promoter=promoter,
            pdf_filename=filename,
            status="Processing",
            toc_verified=True  # ✅ TOC is assumed valid
        )
        db.session.add(entry)
        db.session.commit()

        # Step 4: Update index JSON
        index_path = r"C:\Users\2000166072\Desktop\dhrp2\dhrp_index.json"
        new_index_entry = {
            "company": company.strip(),
            "bse_code": bse_code.strip(),
            "upload_date": upload_date.strip(),
            "uploader_name": uploader_name.strip(),
            "promoter": promoter.strip(),
            "pdf_filename": filename.strip().lower(),
            "status": "Processing",
            "toc_verified": True
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

        # Step 5: Load TOC from pre-generated file
        toc_path = os.path.join('toc', f"{base}.json")
        if not os.path.exists(toc_path):
            logging.warning(f"⚠️ TOC file missing: {toc_path}")
            return jsonify({"success": False, "message": "TOC file missing. Cannot proceed."}), 400

        with open(toc_path, "r", encoding="utf-8") as f:
            hierarchy = json.load(f)

        save_toc_to_db(base, hierarchy)
        logging.info(f"📑 TOC loaded and saved to DB: {toc_path}")

        # Step 6: Launch background processing thread
        def run_background_process(base, entry):
            with app.app_context():
                background_process_dhrp(base, entry)

        thread = threading.Thread(target=run_background_process, args=(base, entry))
        thread.start()
        logging.info(f"🚀 Background processing started for: {filename}")

        # Step 7: Return response
        return jsonify({
            "success": True,
            "message": "Upload and processing started",
            "base": base,
            "entry": {
                "company": company,
                "bse_code": bse_code,
                "upload_date": upload_date,
                "uploader_name": uploader_name,
                "promoter": promoter,
                "pdf_filename": filename,
                "status": "Processing"
            }
        }), 202

    except Exception as e:
        logging.error(f"❌ Demo upload error: {str(e)}")
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
        logging.info(f"🔧 Chunking PDF by TOC: {pdf_path}")
        chunk_pdf_by_toc(pdf_path, toc_path, raw_pkl_path)

        update_processing_stage(base, "🔗 Embedding chunks")
        logging.info(f"🔗 Embedding chunks: {raw_pkl_path}")
        embed_chunks_optimal(raw_pkl_path, embedded_pkl_path)

        update_processing_stage(base, "📦 Loading embedded chunks")
        if not os.path.exists(embedded_pkl_path):
            raise FileNotFoundError(f"Embedded chunks not found: {embedded_pkl_path}")
        with open(embedded_pkl_path, 'rb') as f:
            embedded_chunks = pickle.load(f)

        update_processing_stage(base, "🧠 Generating risk summary")
        logging.info(f"🧠 Generating risk summary for: {filename}")
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
        logging.info(f"✅ Risk summary saved: {risk_path}")

        update_processing_stage(base, "📊 Evaluating Q&A CSV")
        with open(index_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        matched_entry = next(
            (e for e in entries if e.get("pdf_filename", "").strip().lower() == source_pdf_filename),
            None
        )
        if not matched_entry:
            raise ValueError(f"No entry found in index for PDF: {source_pdf_filename}")

        process_csv_and_evaluate(
            csv_path=csv_path,
            output_path=answered_csv_path,
            chunks=embedded_chunks
        )
        logging.info(f"✅ CSV evaluation complete: {answered_csv_path}")

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
            logging.info(f"✅ DHRP status updated to 'Active' in DB for: {filename}")
        else:
            logging.warning(f"⚠️ Could not update status — entry not found for: {filename}")

        update_processing_stage(base, "✅ Completed")
        logging.info(f"📈 Status updated to 'Completed' for: {filename}")
    except Exception as e:
        update_processing_stage(base, "❌ Error during processing")
        logging.error(f"❌ Error during background processing for {base}: {str(e)}")