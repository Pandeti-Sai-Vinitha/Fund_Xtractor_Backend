from flask import Blueprint, request, jsonify
import logging
from services.csv_service import analyze_csv_file, view_csv_file

csv_bp = Blueprint("csv", __name__)

@csv_bp.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    try:
        selected_doc = request.form.get('selected_doc')
        csv_file = request.files.get('csv')
        success, result = analyze_csv_file(selected_doc, csv_file)
        if success:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    except Exception as e:
        logging.error(f"❌ Error during CSV analysis for {selected_doc}: {str(e)}")
        return jsonify({"error": f"Failed to analyze CSV: {str(e)}"}), 500

@csv_bp.route('/view_csv/<filename>')
def view_csv(filename):
    try:
        success, result = view_csv_file(filename)
        if success:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
    except Exception as e:
        logging.error(f"❌ Error reading CSV {filename}: {str(e)}")
        return jsonify({"success": False, "message": f"❌ Failed to read CSV: {str(e)}"}), 500
