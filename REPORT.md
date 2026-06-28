PDF RAG Assistant - Development Report
Project Goal

The goal of this project is to build a local PDF-based question answering assistant. The application will read a PDF document, split its text into smaller chunks, create embeddings for semantic search, retrieve the most relevant chunks for a user question, and generate an answer based on the retrieved content.

The first sample document is a chess knowledge handbook. This document was chosen because it is text-based, easy to understand, and suitable for testing question-answering performance.

Planned Pipeline

The project will follow this pipeline:

1-Extract text from a PDF file.
2-Split the extracted text into smaller chunks.
3-Generate local embeddings for each chunk.
4-Retrieve the most relevant chunks for a user question.
5-Generate an answer using the retrieved context.
6-Build a simple interface for asking questions.
7-Later, add support for user-uploaded PDF files.
Step 1: Project Setup

I created the initial project structure with separate folders for source code and data.

Current structure:

pdf-rag-assistant/
  data/
    chess_knowledge_handbook_rag_sample.pdf
  src/
  README.md
  REPORT.md
  requirements.txt
  .gitignore

I also initialized a Git repository and connected it to GitHub. This allows me to track the development process step by step.

Step 2: Virtual Environment

I created a virtual environment for the project. The purpose of using a virtual environment is to keep the dependencies of this project separate from other Python projects on my computer.

Command used:

python -m venv venv

Then I activated it on Windows:

venv\Scripts\activate
Step 3: PDF Text Extraction

The first technical goal is to extract text from the PDF file page by page. For this step, I used the pypdf library.

The expected output of this step is a list of pages, where each page contains:

the page number
the extracted text

This is important because later the assistant should be able to show which page was used as the source for an answer.

Example structure:

[
    {"page": 1, "text": "..."},
    {"page": 2, "text": "..."}
]
Notes and Observations
The project works best with text-based PDFs.
Scanned PDFs may not work properly because they require OCR. Maybe I 'll add it later. 
Keeping page numbers is important for source-grounded answers.
The first version will use a fixed PDF file. (chess_handbook pdf)
Later versions will support user-uploaded PDFs.

pip freeze > requirements.txt --> this comment provides that using python packages is written in requirements.txt  .
If somebody download my this project from github he can install same python packages with : pip install -r requirements.txt

___________________________________________________
## Step 3: PDF Text Extraction

In this step, I implemented the first part of the RAG pipeline: extracting text from a PDF document.

I used the `pypdf` library because it allows reading text-based PDF files page by page.

The function `extract_text_from_pdf` takes a PDF path as input and returns a list of dictionaries. Each dictionary contains the page number and the extracted text.

This page-level structure is important because later the assistant should show source pages when answering user questions.

Example output structure:

```python
[
    {"page": 1, "text": "..."},
    {"page": 2, "text": "..."}
]
___________________________________________________
## Step 4: Text Chunking

In this step, I implemented text chunking. The extracted page-level text is split into smaller overlapping chunks.

Chunking is necessary because the full PDF text can be too long to search or send directly to a language model. Instead, the system will search over smaller pieces of text and retrieve only the most relevant ones.

Each chunk contains:

- a unique chunk id
- the source page number
- the chunk text

I used a character-based chunking approach with:

- `chunk_size = 900`
- `overlap = 150`  these numbers are easily changeable depending situations.

The overlap helps prevent important information from being lost when a sentence or topic is split between two chunks.

Example chunk structure:

```python
{
    "chunk_id": 0,
    "page": 1,
    "text": "..."
}



Step 5: Local Embeddings

In this step, I generated local embeddings for the text chunks.

After extracting text from the PDF and splitting it into smaller chunks, the next goal was to convert each chunk into a numerical vector. These vectors are called embeddings.

An embedding is a numerical representation of text. Instead of comparing only exact words, embeddings allow the system to compare the meanings of texts.

For example:

"What is castling?"

and

"Castling is a special move involving the king and rook."

may not be exactly the same sentence, but they are semantically related. By converting both into vectors, the system can detect that they are close in meaning.

Installed Packages

For this step, I installed:

pip install sentence-transformers numpy

Then I updated the dependency file with:

pip freeze > requirements.txt

This command writes the installed Python packages and their versions into requirements.txt.

If someone downloads this project from GitHub, they can install the same dependencies with:

pip install -r requirements.txt

The requirements.txt file became longer than expected because sentence-transformers automatically installs several dependency packages such as torch, transformers, huggingface_hub, scikit-learn, scipy, and tokenizers.

Embedding Model

I used the following local embedding model:

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

This model was chosen because it is small, fast, free, and suitable for semantic search. It converts each text into a 384-dimensional vector.

The first time I ran the code, the model was downloaded from Hugging Face. This made the first run take around 30 seconds. After the first download, the model is cached locally, so future runs should be faster.

What the Embedding Code Does

The embedding code follows this process:

Load the sample PDF.
Extract text from the PDF page by page.
Split the extracted text into chunks.
Load the local embedding model.
Convert each chunk into an embedding vector.
Store all vectors in a NumPy array.

The main function is:

def create_embeddings(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:

This function takes the list of text chunks and the embedding model as input. It extracts only the "text" field from each chunk and passes those texts to the model.

Each chunk originally looks like this:

{
    "chunk_id": 0,
    "page": 1,
    "text": "Chess is a board game..."
}

The embedding model only needs the text part:

texts = [chunk["text"] for chunk in chunks]

Then the model converts these texts into vectors:

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)
NumPy Array

The result is converted into a NumPy array:

return np.array(embeddings)

A NumPy array is useful because embeddings are numerical vectors, and later I will need to perform mathematical operations on them.

For example, in the retrieval step, the system will compare the user question vector with all chunk vectors. NumPy makes this kind of vector comparison easier and faster.

Normalized Embeddings

I used:

normalize_embeddings=True

This normalizes the vectors. Normalized vectors make similarity comparison easier because the dot product can be used like cosine similarity.

This will be important in the next step, where I retrieve the most relevant chunks for a user question.

Output

When I ran:

python src/embeddings.py

I got the following important output:

Total chunks: 46
Embeddings shape: (46, 384)

This means:

The PDF was split into 46 chunks.
Each chunk was converted into a 384-dimensional vector.

I also printed the first 10 numbers of the first embedding vector to confirm that the model successfully produced numerical vectors.

Example:

First embedding vector preview:
[-0.00349292  0.06987692 -0.03980904 -0.0893418   0.02732148
  0.06170318 -0.01014277  0.10034266 -0.0067331   0.06754159]

The individual numbers are not meaningful by themselves. What matters is that each chunk now has a numerical representation that can be compared with user questions.

Notes and Questions I Clarified

During this step, I clarified the following concepts:

A vector is a list of numbers.
An embedding is a vector representation of text.
Similar meanings should have closer vectors.
A NumPy array is useful for storing and comparing many vectors.
The shape (46, 384) means 46 chunks and 384 numbers per chunk.
The first model download can take longer, but later runs should be faster.
The Hugging Face symlink warning on Windows is not a critical issue. It only means caching may use slightly more disk space.


## Step 6: Semantic Retrieval

In this step, I implemented semantic retrieval.

The goal of semantic retrieval is to find the most relevant PDF chunks for a user question.

The process is:

1. Convert the user question into an embedding vector.
2. Compare the question embedding with all chunk embeddings.
3. Calculate similarity scores.
4. Sort chunks by similarity score.
5. Return the top-k most relevant chunks.

Because the chunk embeddings and question embedding are normalized, I used dot product as a cosine similarity-like comparison.

Example:

```text
Question: What is castling?

Top retrieved chunks:
1. Page 2 - similarity score: ...
2. Page 3 - similarity score: ...