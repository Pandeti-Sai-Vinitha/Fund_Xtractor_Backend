from flask import Blueprint, jsonify
import logging

from models.dhrp_entry import DhrpEntry


dhrp_bp = Blueprint("dhrps", __name__)

@dhrp_bp.route('/get_all_dhrps')
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
