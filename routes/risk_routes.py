from flask import Blueprint, jsonify
import os, json, logging
from models.dhrp_entry import DhrpEntry, RiskSummary
from utils.helpers import normalize_name, get_base_name


risk_bp = Blueprint("risk", __name__)

@risk_bp.route('/risk/<doc>')
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
