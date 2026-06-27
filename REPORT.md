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
___________________________________________________