from flask import Blueprint, jsonify
import logging

from app import db
from models.dhrp_entry import DhrpEntry

toc_verify_bp = Blueprint("toc_verify", __name__)

@toc_verify_bp.route('/verify_toc/<base>', methods=['POST'])
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

@toc_verify_bp.route('/accept_toc/<base>', methods=['POST'])
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
