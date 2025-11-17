from flask import Blueprint, request, jsonify
import logging
from services.upload_csv_service import handle_csv_upload

upload_csv_bp = Blueprint("upload_csv", __name__)

@upload_csv_bp.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        success, result = handle_csv_upload(request)
        return jsonify(result), (200 if success else 400)
    except Exception as e:
        logging.error(f"❌ Error during CSV upload: {str(e)}")
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'}), 500
