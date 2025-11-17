from flask import Blueprint, jsonify
import os, json

from models.dhrp_entry import DhrpEntry, ProcessingStatus


status_bp = Blueprint("status", __name__)

@status_bp.route('/stream-status/<base>')
def get_status(base):
    status_path = os.path.join('status', f"{base}.json")
    if not os.path.exists(status_path):
        return jsonify({"success": False, "message": "Status not available yet."}), 200
    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        return jsonify({"success": True, "base": base, "status": status_data})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading status file: {str(e)}"}), 500

@status_bp.route('/status/<base>', methods=['GET'])
def get_processing_status(base):
    entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
    if not entry:
        return jsonify({"success": False, "message": "Entry not found"}), 404
    status = ProcessingStatus.query.filter_by(dhrp_id=entry.id).first()
    if not status:
        return jsonify({"success": False, "message": "No status found"}), 404
    return jsonify({"success": True, "stage": status.processing_stage, "updated_at": status.updated_at})
