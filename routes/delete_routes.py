from flask import Blueprint, jsonify
import logging
from services.delete_service import delete_document

delete_bp = Blueprint("delete", __name__)

@delete_bp.route('/delete/<doc>', methods=['POST'])
def delete_doc(doc):
    try:
        success, message = delete_document(doc)
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"message": message}), 404
    except Exception as e:
        logging.error(f"❌ Error deleting document {doc}: {str(e)}")
        return jsonify({"message": f"Error deleting {doc}: {str(e)}"}), 500
