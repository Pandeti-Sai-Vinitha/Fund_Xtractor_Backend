import pickle

def view_embedded_chunks(pkl_path, max_preview_chars=300, max_chunks=10):
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"\n📂 Loaded {len(chunks)} embedded chunks from {pkl_path}\n")

    missing = sum(1 for c in chunks if 'embedding_vector' not in c)
    print(f"❌ Missing embeddings in {missing} chunks\n")

    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f"🔹 Chunk {i + 1}/{len(chunks)}")
        print(f"  ID         : {chunk.get('chunk_id', '—')}")
        print(f"  Section    : {chunk.get('section', '—')}")
        print(f"  Subsection : {chunk.get('subsection', '—')}")
        print(f"  Pages      : {chunk.get('start_page', '—')}–{chunk.get('end_page', '—')} (PDF {chunk.get('pdf_start_page', '—')}–{chunk.get('pdf_end_page', '—')})")
        print(f"  Characters : {chunk.get('char_count', '—')}")
        if 'embedding_vector' in chunk:
            print(f"  Embedding  : {len(chunk['embedding_vector'])} dims")
        else:
            print(f"  Embedding  : ❌ Not available")
        preview = chunk.get('text', '').strip().replace('\n', ' ')[:max_preview_chars]
        print(f"  Preview    : {preview}")
        print("-" * 80)

if __name__ == "__main__":
    pkl_path = "pickles/tata_capital_embedded.pkl"
    view_embedded_chunks(pkl_path)
