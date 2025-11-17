import pickle
import re
from keybert import KeyBERT

kw_model = KeyBERT('all-MiniLM-L6-v2')

def suggest_keywords(chunk, max_keywords=5):
    section = chunk.get("section_title", "")
    subsection = chunk.get("subsection_title", "")
    combined_title = f"{section} {subsection}".strip().upper()

    title_words = re.findall(r'\b[A-Z][A-Z]+\b', combined_title)
    title_keywords = [word.lower() for word in title_words if word not in {"SECTION", "OF", "AND", "THE", "OUR"}]

    text = chunk.get("text", "")
    bert_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=max_keywords)
    bert_keywords = [kw for kw, _ in bert_keywords]

    combined = list(set(title_keywords + bert_keywords))
    return combined if combined else ["general", "information", "document"]

def save_chunks_to_pkl(chunks, output_path="dhrp_chunks.pkl"):
    with open(output_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ Saved {len(chunks)} chunks to {output_path}")

def view_chunks_from_pkl(pkl_path="dhrp_chunks.pkl"):
    with open(pkl_path, 'rb') as f:
        chunks = pickle.load(f)
    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Chunk {i+1}")
        print(f"Section: {chunk.get('section_title')}")
        if 'subsection_title' in chunk:
            print(f"Subsection: {chunk['subsection_title']}")
        print(f"Pages: {chunk['start_page']}–{chunk['end_page']}")
        print(f"Text Preview: {chunk['text'][:300]}...")
        print(f"Embedding shape: {chunk['embedding'].shape}")
        print(f"Suggested Keywords: {suggest_keywords(chunk)}")
    return chunks
