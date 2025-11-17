import logging
from utils.helpers import load_index

def get_company_details(doc):
    entries = load_index()
    for entry in entries:
        if entry.get('pdf_filename') == doc:
            logging.info(f"🏢 Company details retrieved for: {doc} — Company: {entry.get('company', 'Unknown')}")
            return True, entry
    logging.warning(f"⚠️ Company details not found for: {doc}")
    return False, "Company details not found"
