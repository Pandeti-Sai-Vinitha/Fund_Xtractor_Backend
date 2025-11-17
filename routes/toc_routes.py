from flask import Blueprint, jsonify
import os, json
from models.dhrp_entry import DhrpEntry, TocSection
from utils.helpers import normalize_name, get_base_name

toc_bp = Blueprint("toc", __name__)

@toc_bp.route('/get_toc/<doc>')
def get_toc(doc):
    try:
        base = normalize_name(get_base_name(doc))
        toc_path = os.path.join('toc', f"{base}.json")

        # Try JSON file first
        if os.path.exists(toc_path):
            try:
                with open(toc_path, 'r', encoding='utf-8') as f:
                    toc_data = json.load(f)
                toc_dict = toc_data.get("hierarchy", toc_data)
                if toc_dict:
                    toc_list = []
                    for title, data in toc_dict.items():
                        page_num = data.get("pdf_start_page") or data.get("start_page") or data.get("page")
                        if isinstance(page_num, str):
                            try:
                                page_num = int(page_num)
                            except Exception:
                                page_num = None
                        subs_list = []
                        subs = data.get("subsections", {})
                        if isinstance(subs, dict):
                            for sub_title, sub_data in subs.items():
                                sub_page = sub_data.get("pdf_start_page") or sub_data.get("start_page")
                                if isinstance(sub_page, str):
                                    try:
                                        sub_page = int(sub_page)
                                    except Exception:
                                        sub_page = None
                                subs_list.append({"title": sub_title, "page": sub_page})
                        toc_list.append({"title": title, "page": page_num, "subsections": subs_list})
                    return jsonify({"toc": toc_list})
            except Exception:
                pass

        # Fallback to DB
        entry = DhrpEntry.query.filter_by(pdf_filename=f"{base}.pdf").first()
        if not entry:
            return jsonify({"toc": []})
        sections = TocSection.query.filter_by(dhrp_id=entry.id).all()
        if not sections:
            return jsonify({"toc": []})

        toc_dict = {}
        for section in sections:
            title = section.title
            if title not in toc_dict:
                toc_dict[title] = {"page": section.page, "subsections": []}
            if section.subsection_title:
                toc_dict[title]["subsections"].append({
                    "title": section.subsection_title,
                    "page": section.subsection_page
                })

        toc_list = [{"title": t, "page": d["page"], "subsections": d["subsections"]} for t, d in toc_dict.items()]
        return jsonify({"toc": toc_list})

    except Exception:
        return jsonify({"toc": []}), 500
