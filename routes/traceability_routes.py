from flask import Blueprint, jsonify
from services.traceability_service import get_traceability_data

trace_bp = Blueprint("traceability", __name__)

@trace_bp.route('/get_traceability/<company>')
def get_traceability(company):
    return jsonify(get_traceability_data(company))
