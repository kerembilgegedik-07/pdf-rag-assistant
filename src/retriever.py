import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_reader import extract_text_from_pdf
from chunker import create_chunks
from embeddings import load_embedding_model, create_embeddings


def create_question_embedding(question: str, model: SentenceTransformer) -> np.ndarray:
    """
    Convert the user's question into an embedding vector.

    The question must be represented in the same vector space as the PDF chunks.
    This allows us to compare the question with each chunk.

    Example:
        "What is castling?"
        -> [0.12, -0.03, 0.44, ..., 0.08]
    """
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    return np.array(question_embedding)


def retrieve_top_chunks(
    question: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a user question.

    Args:
        question:
            The user's question.

        chunks:
            The original text chunks.
            Each chunk contains chunk_id, page, and text.

        chunk_embeddings:
            NumPy array containing the embedding vector of each chunk.
            The order of chunk_embeddings must match the order of chunks.

        model:
            The same embedding model used for creating chunk embeddings.

        top_k:
            Number of most relevant chunks to return.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - chunk_id
        - page
        - text
        - similarity_score
    """

    # Step 1: Convert the user question into an embedding vector.
    question_embedding = create_question_embedding(question, model)

    # Step 2: Calculate similarity between the question vector
    # and all chunk vectors.
    #
    # Because we used normalize_embeddings=True, dot product works like
    # cosine similarity.
    #
    # Result example:
    # similarities = [0.12, 0.85, 0.43, 0.21, ...]
    #
    # Each score corresponds to one chunk:
    # similarities[0] -> chunks[0]
    # similarities[1] -> chunks[1]
    similarities = chunk_embeddings @ question_embedding

    # Step 3: Sort chunks by similarity score from highest to lowest.
    #
    # np.argsort(similarities) gives indices in ascending order.
    # [::-1] reverses it, so highest scores come first.
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Step 4: Build a result list with chunk information and score.
    results = []

    for index in top_indices:
        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "similarity_score": float(similarities[index])
        })

    return results


if __name__ == "__main__":
    pdf_path = "data/chess_knowledge_handbook_rag_sample.pdf"

    # Step 1: Extract text from PDF.
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Split PDF text into chunks.
    chunks = create_chunks(pages)

    # Step 3: Load embedding model.
    model = load_embedding_model()

    # Step 4: Create embeddings for all chunks.
    chunk_embeddings = create_embeddings(chunks, model)

    # Step 5: Ask a test question.
    question = "What is castling?"

    # Step 6: Retrieve the most relevant chunks.
    top_chunks = retrieve_top_chunks(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        model=model,
        top_k=5
    )

    print(f"Question: {question}")
    print("-" * 80)

    for rank, chunk in enumerate(top_chunks, start=1):
        print(f"Rank: {rank}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']}")
        print(f"Similarity Score: {chunk['similarity_score']:.4f}")
        print("Text Preview:")
        print(chunk["text"][:700])
        print("-" * 80)