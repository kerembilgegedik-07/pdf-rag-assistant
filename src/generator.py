from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pdf_reader import extract_text_from_pdf
from chunker import create_chunks
from embeddings import load_embedding_model, create_embeddings
from retriever import retrieve_top_chunks


GENERATOR_MODEL_NAME = "google/flan-t5-small"


def load_generator_model(model_name: str = GENERATOR_MODEL_NAME):
    """
    Load a local text generation model.

    This model will generate an answer using the retrieved PDF chunks.
    The first run may take longer because the model is downloaded.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def build_context(retrieved_chunks: list[dict], max_context_chars: int = 2500) -> str:
    """
    Build a single context string from the retrieved chunks.

    The generator model should not receive the entire PDF.
    It should only receive the most relevant chunks found by the retriever.

    Args:
        retrieved_chunks:
            The top chunks returned by semantic retrieval.

        max_context_chars:
            Maximum number of characters to include in the context.
            This prevents the prompt from becoming too long.

    Returns:
        A formatted context string.
    """
    context_parts = []
    current_length = 0

    for chunk in retrieved_chunks:
        chunk_text = f"[Page {chunk['page']}, Chunk {chunk['chunk_id']}]\n{chunk['text']}\n"

        if current_length + len(chunk_text) > max_context_chars:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    tokenizer,
    model,
    max_new_tokens: int = 180
) -> str:
    """
    Generate an answer using only the retrieved chunks as context.

    The prompt tells the model not to use outside knowledge.
    If the answer is not in the context, the model should say that
    it could not find the answer in the document.
    """
    context = build_context(retrieved_chunks)

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say: "I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def get_source_pages(retrieved_chunks: list[dict]) -> list[int]:
    """
    Return unique page numbers used as sources.
    """
    pages = []

    for chunk in retrieved_chunks:
        if chunk["page"] not in pages:
            pages.append(chunk["page"])

    return pages


if __name__ == "__main__":
    pdf_path = "data/chess_knowledge_handbook_rag_sample.pdf"

    # Step 1: Extract text from the PDF.
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Split extracted text into chunks.
    chunks = create_chunks(pages)

    # Step 3: Load embedding model and create chunk embeddings.
    embedding_model = load_embedding_model()
    chunk_embeddings = create_embeddings(chunks, embedding_model)

    # Step 4: Test question.
    question = "What is castling?"

    # Step 5: Retrieve the most relevant chunks for the question.
    retrieved_chunks = retrieve_top_chunks(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        model=embedding_model,
        top_k=5
    )

    # Step 6: Load local answer generation model.
    tokenizer, generator_model = load_generator_model()

    # Step 7: Generate answer using retrieved chunks.
    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        tokenizer=tokenizer,
        model=generator_model
    )

    source_pages = get_source_pages(retrieved_chunks)

    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    print("Answer:")
    print(answer)
    print("-" * 80)
    print(f"Source pages: {source_pages}")
    print("=" * 80)

    print("\nRetrieved chunks used as context:")
    print("-" * 80)

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        print(f"Rank: {rank}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Similarity Score: {chunk['similarity_score']:.4f}")
        print(chunk["text"][:500])
        print("-" * 80)