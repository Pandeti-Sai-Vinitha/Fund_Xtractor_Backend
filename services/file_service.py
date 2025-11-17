import os, logging
from werkzeug.utils import secure_filename
from utils.helpers import normalize_name, get_base_name
from flask import current_app as app

def save_pdf(pdf, uploader_name, company):
    original_filename = secure_filename(pdf.filename)
    base = normalize_name(get_base_name(original_filename))
    filename = f"{base}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(pdf_path)
    logging.info(f"📄 PDF saved: {filename} by {uploader_name} for {company}")
    return filename, pdf_path, base
