import logging

from app import db
from models.dhrp_entry import DhrpEntry

def save_entry(company, bse_code, upload_date, uploader_name, promoter, filename):
    entry = DhrpEntry(
        company=company,
        bse_code=bse_code,
        upload_date=upload_date,
        uploader_name=uploader_name,
        promoter=promoter,
        pdf_filename=filename,
        status="Processing"
    )
    db.session.add(entry)
    db.session.commit()
    logging.info(f"📥 DHRP entry saved to database: {filename}")
