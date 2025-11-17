from flask import Blueprint, jsonify
import os, logging, threading

from app import db, app
from models.dhrp_entry import DhrpEntry
from services.process_service import background_process_dhrp

process_bp = Blueprint("process", __name__)

@process_bp.route('/process/<base>', methods=['POST'])
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
