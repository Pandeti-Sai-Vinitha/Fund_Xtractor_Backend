from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from utils.helpers import normalize_name, get_base_name

toc_replace_bp = Blueprint("toc_replace", __name__)

@toc_replace_bp.route('/api/replace_toc_file', methods=['POST'])
def replace_toc_file():
    uploaded_file = request.files.get('file')
    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    
    filename = secure_filename(uploaded_file.filename)
    base_name = normalize_name(get_base_name(filename))
    os.makedirs('toc', exist_ok=True)
    
    if filename.endswith('.json'):
        json_path = os.path.join('toc', f"{base_name}.json")
        uploaded_file.save(json_path)
        return jsonify({"success": True, "message": f"TOC JSON replaced for {base_name}", "path": json_path})
    elif filename.endswith('.csv'):
        csv_path = os.path.join('toc', f"{base_name}_edited_toc.csv")
        uploaded_file.save(csv_path)
        return jsonify({"success": True, "message": f"TOC CSV replaced for {base_name}", "path": csv_path})
    else:
        return jsonify({"success": False, "message": "Only .json or .csv files are accepted"}), 400
