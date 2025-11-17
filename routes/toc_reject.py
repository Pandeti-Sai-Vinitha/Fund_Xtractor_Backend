from flask import Blueprint, jsonify
import os, logging

from app import db
from models.dhrp_entry import DhrpEntry, ProcessingStatus, RiskSummary, TocSection

toc_reject_bp = Blueprint("toc_reject", __name__)

@toc_reject_bp.route('/reject_toc/<base>', methods=['POST'])
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
