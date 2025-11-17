from flask import Blueprint, jsonify, request
import os, json, logging, pandas as pd
from PyPDF2 import PdfReader
from utils.helpers import normalize_hierarchy_ranges, get_base_name, normalize_name
from services import toc_service

toc_save_bp = Blueprint("toc_save", __name__)

@toc_save_bp.route('/save_toc/<doc>')
def save_toc(doc):
    """
    Extract TOC directly from a PDF using toc_service functions.
    """
    try:
        base = normalize_name(get_base_name(doc))
        pdf_path = os.path.join("uploads", f"{base}.pdf")

        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF not found for {doc}"}), 404

        pdf_reader = PdfReader(pdf_path)

        # Run extraction pipeline
        toc, toc_page_number, _ = toc_service.extract_and_correct_toc(pdf_path, page_offset=0)

        total_pages = len(pdf_reader.pages)
        toc = toc_service.correct_end_pages(toc, total_pages)

        # Normalize hierarchy ranges
        hierarchy = normalize_hierarchy_ranges(toc, total_pages=total_pages)

        # Save TOC JSON
        toc_path = os.path.join("toc", f"{base}.json")
        os.makedirs("toc", exist_ok=True)
        with open(toc_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2)

        logging.info(f"📑 TOC extracted and saved for {doc} at {toc_path}")

        return jsonify({"success": True, "toc": hierarchy, "toc_file": toc_path})
    except Exception as e:
        logging.error(f"❌ Failed to extract TOC for {doc}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@toc_save_bp.route('/api/save_toc', methods=['POST'])
def save_edited_toc():
    """
    Accept edited TOC rows from frontend, convert to DataFrame, 
    and rebuild hierarchy using toc_service.toc_to_dataframe.
    """
    try:
        data = request.get_json()
        pdf_name = data.get("pdfName")
        rows = data.get("rows", [])

        if not pdf_name or not rows:
            return jsonify({"error": "Missing PDF name or TOC rows"}), 400

        base = normalize_name(get_base_name(pdf_name))
        csv_path = os.path.join("toc", f"{base}_edited.csv")
        json_path = os.path.join("toc", f"{base}.json")

        # Save edited rows to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Convert DataFrame back to hierarchy
        hierarchy = toc_service.toc_to_dataframe(df)

        # Save hierarchy JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2)

        logging.info(f"✅ Edited TOC saved for {pdf_name} at {json_path}")

        return jsonify({"success": True, "toc": hierarchy, "csv_file": csv_path, "json_file": json_path})
    except Exception as e:
        logging.error(f"❌ Failed to save edited TOC: {str(e)}")
        return jsonify({"error": str(e)}), 500
