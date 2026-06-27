

from pypdf import PdfReader   # it reads pdf and gives us object 


def extract_text_from_pdf(pdf_path: str) -> list[dict]:   # parameter is path of pdf
    """
    Extracts text from a PDF file page by page.

    Returns:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."}
        ]
    """
    reader = PdfReader(pdf_path)
    pages = []     # read all pages of pdf and add this list. 

    for index, page in enumerate(reader.pages):   # index , page object    which is provided by pdf reader
        text = page.extract_text()

        if text is None:
            text = ""

        pages.append({
            "page": index + 1,
            "text": text.strip()     #delete unnecessary spaces.
        })

    return pages


if __name__ == "__main__":      # This block runs only when this file is executed directly.
                                # It is used for testing the PDF text extraction function.
    pdf_path = "data/chess_knowledge_handbook_rag_sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Total pages: {len(pages)}")
    print("-" * 60)

    for page in pages[:2]:
        print(f"PAGE {page['page']}")
        print(page["text"][:1200])
        print("-" * 60)