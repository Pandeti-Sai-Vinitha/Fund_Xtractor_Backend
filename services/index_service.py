import os
import json
import logging

BASE_DIR = os.getcwd()
INDEX_FILE = os.path.join(BASE_DIR, "dhrp_index.json")

def load_index():
    """Load existing index entries from JSON file."""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_index(entries):
    """Save index entries back to JSON file."""
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

def update_index(company, bse_code, upload_date, uploader_name, promoter, filename):
    """
    Add a new entry to the DHRP index JSON file.
    """
    new_index_entry = {
        "company": company.strip(),
        "bse_code": bse_code.strip(),
        "upload_date": upload_date.strip(),
        "uploader_name": uploader_name.strip(),
        "promoter": promoter.strip(),
        "pdf_filename": filename.strip().lower(),
        "status": "New",
        "toc_verified": False
    }

    try:
        entries = load_index()
        entries.append(new_index_entry)
        save_index(entries)
        logging.info(f"🗂️ Entry added to index: {filename}")
        return True, new_index_entry
    except Exception as e:
        logging.warning(f"⚠️ Failed to update index for {filename}: {str(e)}")
        return False, str(e)
