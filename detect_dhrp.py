import fitz

def is_dhrp(pdf_path, max_pages=5):
    doc = fitz.open(pdf_path)
    keywords = [
        "draft red herring prospectus",
        "summary of the offer",
        "risk factors",
        "offer details",
        "covenants",
        "capital structure",
        "basis for offer price",
        "statement on special tax benefits",
        "industry overview",
        "restated financial information",
        "management discussion and analysis",
        "legal and other information"
    ]

    found = set()
    for page_num in range(min(max_pages, len(doc))):
        text = doc.load_page(page_num).get_text().lower()
        for keyword in keywords:
            if keyword in text:
                found.add(keyword)

    return len(found) >= 3
