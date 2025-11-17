from flask import Blueprint, jsonify, send_file
import logging
from services.download_service import get_download_file

download_bp = Blueprint("download", __name__)

@download_bp.route('/download/<filename>')
def download_csv(filename):
    try:
        success, result = get_download_file(filename)
        if success:
            return send_file(result, as_attachment=True)
        else:
            return jsonify({"error": result}), 404
    except Exception as e:
        logging.error(f"❌ Error during file download {filename}: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500
