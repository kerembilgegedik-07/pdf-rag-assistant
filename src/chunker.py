from pdf_reader import extract_text_from_pdf


def create_chunks(
    pages: list[dict],
    chunk_size: int = 900,
    overlap: int = 150    #overlap is cruical 
) -> list[dict]:
    """
    Split page-level PDF text into smaller overlapping chunks.

    Args:
        pages: A list of dictionaries returned by extract_text_from_pdf.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Number of characters repeated between consecutive chunks.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - chunk_id
        - page number
        - chunk text
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    chunk_id = 0

    for page in pages:
        page_number = page["page"]

        # Normalize whitespace so the text becomes cleaner.
        text = " ".join(page["text"].split())   # Chess     is   a      board game. -->Chess is a board game.
                                                #so it gets cleaner.
        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk_text
            })

            chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap    # for example second chunk starts at 150.

    return chunks


if __name__ == "__main__":
    pdf_path = "data/chess_knowledge_handbook_rag_sample.pdf"

    pages = extract_text_from_pdf(pdf_path)
    chunks = create_chunks(pages)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print("-" * 60)

    for chunk in chunks[:3]:
        print(f"CHUNK ID: {chunk['chunk_id']}")
        print(f"PAGE: {chunk['page']}")
        print(f"LENGTH: {len(chunk['text'])} characters")
        print(chunk["text"][:500])
        print("-" * 60)