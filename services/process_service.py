import os, json, logging, pickle, pandas as pd
from app import db, app


from models.dhrp_entry import DhrpEntry

from services.chunker import chunk_pdf_by_toc
from services.embedder import embed_chunks_optimal
from services.risk_summary import generate_risk_summary_from_chunks
from services.stream_csv_service import stream_csv_evaluation
from utils.helpers import update_processing_stage, save_index, save_risk_summary_to_db


BASE_DIR = os.getcwd()

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

        # Validate inputs
        for label, path in [("PDF", pdf_path), ("TOC", toc_path)]:
            if not path or not isinstance(path, str):
                raise ValueError(f"{label} path is invalid: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        # Ensure CSV template exists
        if not os.path.exists(csv_path):
            logging.warning(f"⚠️ CSV template missing — creating fallback at: {csv_path}")
            fallback_questions = [
                "What is the business model?",
                "What are the key financial risks?",
                "Does the company meet profitability criteria?"
            ]
            pd.DataFrame({"Question": fallback_questions}).to_csv(csv_path, index=False)

        # Processing pipeline
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
        risk_text, summary_bullets, _ = generate_risk_summary_from_chunks(
            chunks=embedded_chunks,
            user_question="What are the key risk factors disclosed in the DHRP?",
            search_intent="Search 'Risk Factors', 'Business Overview', and 'Management Discussion and Analysis' for risk-related disclosures.",
            remarks="Focus on financial, operational, regulatory, and competitive risks. Include any mitigation strategies or gaps.",
            cache_dir="cache"
        )

        update_processing_stage(base, "📝 Saving risk summary")
        with open(risk_path, 'w', encoding='utf-8') as f:
            json.dump({"summary_bullets": summary_bullets, "risk_text": risk_text}, f, indent=2)
        save_risk_summary_to_db(base, risk_text, summary_bullets)

        update_processing_stage(base, "📊 Evaluating Q&A CSV")
        with open(index_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        matched_entry = next((e for e in entries if e.get("pdf_filename", "").strip().lower() == source_pdf_filename), None)
        if not matched_entry:
            raise ValueError(f"No entry found in index for PDF: {source_pdf_filename}")

        stream_csv_evaluation(base=base, chunks=embedded_chunks, csv_path=csv_path, output_path=answered_csv_path)

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
