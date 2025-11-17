from flask import Blueprint, jsonify
import logging
from services.company_service import get_company_details

company_bp = Blueprint("company", __name__)

@company_bp.route('/get_company/<doc>')
def get_company(doc):
    try:
        success, result = get_company_details(doc)
        if success:
            return jsonify(result)
        else:
            return jsonify({"error": result}), 404
    except Exception as e:
        logging.error(f"❌ Error retrieving company details for {doc}: {str(e)}")
        return jsonify({"error": f"Failed to retrieve company details: {str(e)}"}), 500
