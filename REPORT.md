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