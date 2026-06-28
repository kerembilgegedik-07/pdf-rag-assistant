import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_reader import extract_text_from_pdf
from chunker import create_chunks


# This is the name of the embedding model we will use.
# The model is downloaded from Hugging Face the first time it is used.
# It is small, fast, free, and works locally after being downloaded.
# This model converts each text into a 384-dimensional vector.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the embedding model.

    An embedding model converts text into numerical vectors.

    Example:
        "What is castling?"
        becomes something like:
        [0.12, -0.03, 0.44, ..., 0.08]

    These vectors help us compare the meanings of texts.
    """
    model = SentenceTransformer(model_name)
    return model


def create_embeddings(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Create embeddings for all text chunks.

    Args:
        chunks:
            A list of chunk dictionaries created by create_chunks().
            Each chunk looks like this:

            {
                "chunk_id": 0,
                "page": 1,
                "text": "Chess is a board game..."
            }

        model:
            The SentenceTransformer model that converts text into vectors.

    Returns:
        A NumPy array containing embedding vectors.

        If we have 20 chunks and the model creates 384-dimensional vectors,
        the output shape will be:

        (20, 384)

        This means:
        - 20 rows: one row for each chunk
        - 384 columns: numerical vector values for each chunk
    """

    # The embedding model only needs the text part of each chunk.
    # It does not need chunk_id or page number.
    #
    # Example:
    # chunks = [
    #     {"chunk_id": 0, "page": 1, "text": "Chess is a board game..."},
    #     {"chunk_id": 1, "page": 2, "text": "Castling is a special move..."}
    # ]
    #
    # texts becomes:
    # [
    #     "Chess is a board game...",
    #     "Castling is a special move..."
    # ]
    texts = [chunk["text"] for chunk in chunks]

    # model.encode() converts every text into a numerical vector.
    #
    # normalize_embeddings=True:
    #     This makes all vectors have a standard length.
    #     It helps us compare vectors using cosine similarity later.
    #
    # show_progress_bar=True:
    #     Shows progress while embeddings are being created.
    #embeddings = [
    #[0.03, -0.12, 0.08, ..., 0.19],
    #[-0.22, 0.14, 0.31, ..., -0.02]
    #]
    
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True  #it is useful for long texts
    )

    # Convert the result into a NumPy array.
    # NumPy arrays are useful for mathematical operations,
    # especially when we calculate similarity between question vectors
    # and chunk vectors in the next step.
    return np.array(embeddings)


if __name__ == "__main__":
    # This block runs only when this file is executed directly.
    # Example:
    # python src/embeddings.py
    #
    # If another file imports create_embeddings(),
    # this test block will not run.

    # Path of the sample PDF we use for the first version of the project.
    pdf_path = "data/chess_knowledge_handbook_rag_sample.pdf"

    # Step 1: Extract page-level text from the PDF.
    # pages will look like:
    # [
    #     {"page": 1, "text": "..."},
    #     {"page": 2, "text": "..."}
    # ]
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Split page texts into smaller chunks.
    # chunks will look like:
    # [
    #     {"chunk_id": 0, "page": 1, "text": "..."},
    #     {"chunk_id": 1, "page": 1, "text": "..."}
    # ]
    chunks = create_chunks(pages)

    print(f"Total chunks: {len(chunks)}")

    # Step 3: Load the local embedding model.
    # The first run may take longer because the model may be downloaded.
    model = load_embedding_model()

    # Step 4: Convert all chunks into embedding vectors.
    embeddings = create_embeddings(chunks, model)

    # embeddings.shape tells us the size of the NumPy array.
    #
    # Example output:
    # Embeddings shape: (20, 384)
    #
    # This means:
    # - There are 20 chunks.
    # - Each chunk is represented by a 384-dimensional vector.
    print(f"Embeddings shape: {embeddings.shape}")

    # Print the first 10 numbers of the first chunk's embedding.
    # The individual numbers are not meaningful by themselves.
    # We print them only to confirm that the model produced a vector.
    print("First embedding vector preview:")
    print(embeddings[0][:10])