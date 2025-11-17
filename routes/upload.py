from flask import Blueprint, request, jsonify
import logging
from services import file_service, db_services, index_service, toc_service
from utils.helpers import save_toc_to_db

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload", methods=["POST"])
def upload_dhrp():
    try:
        # Extract form data
        company = request.form.get("company")
        bse_code = request.form.get("bse_code")
        upload_date = request.form.get("upload_date")
        uploader_name = request.form.get("uploader_name")
        promoter = request.form.get("promoter")
        pdf = request.files.get("pdf")

        if not pdf or pdf.filename == "":
            logging.warning(f"Upload failed — No PDF provided by {uploader_name} for {company}")
            return jsonify({"success": False, "message": "No PDF uploaded"}), 400

        # Delegate to services
        filename, pdf_path, base = file_service.save_pdf(pdf, uploader_name, company)
        db_services.save_entry(company, bse_code, upload_date, uploader_name, promoter, filename)
        index_service.update_index(company, bse_code, upload_date, uploader_name, promoter, filename)

        hierarchy = toc_service.extract_and_save_toc(pdf_path, base)
        if hierarchy:
            save_toc_to_db(base, hierarchy)

        return jsonify({
            "success": True,
            "message": "DHRP uploaded successfully",
            "entry": {
                "company": company,
                "bse_code": bse_code,
                "upload_date": upload_date,
                "uploader_name": uploader_name,
                "promoter": promoter,
                "pdf_filename": filename,
                "status": "New"
            },
            "base": base
        }), 200

    except Exception as e:
        logging.error(f"❌ Upload error for {request.form.get('company', 'Unknown')}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500
