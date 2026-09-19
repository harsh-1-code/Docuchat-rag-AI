[README.md](https://github.com/user-attachments/files/32407708/README.md)
# 📚 DocuChat AI --- Chat With Your Documents Using RAG

> A beginner-friendly Retrieval-Augmented Generation (RAG) application
> that lets you upload a PDF, search its content semantically, and ask
> questions about the document using an AI model.

## Table of Contents

-   [What is DocuChat AI?](#what-is-docuchat-ai)
-   [What is RAG?](#what-is-rag)
-   [Features](#features)
-   [Technology Stack](#technology-stack)
-   [How It Works](#how-it-works)
-   [Architecture](#architecture)
-   [Project Structure](#project-structure)
-   [Build From Scratch](#build-from-scratch)
-   [Installation](#installation)
-   [Run Locally](#run-locally)
-   [Understanding the RAG Pipeline](#understanding-the-rag-pipeline)
-   [Deployment](#deployment)
-   [Troubleshooting](#troubleshooting)
-   [Security](#security)
-   [Future Improvements](#future-improvements)
-   [Learning Roadmap](#learning-roadmap)

------------------------------------------------------------------------

## What is DocuChat AI?

DocuChat AI is a document question-answering application built around
Retrieval-Augmented Generation (RAG).

The application allows a user to upload a PDF and ask natural-language
questions about it. The system extracts the PDF text, divides it into
chunks, converts the chunks into embeddings, stores them in a FAISS
vector index, retrieves the most relevant chunks for a question, and
passes that context to an LLM to generate an answer.

The application can also show the source pages associated with retrieved
content.

------------------------------------------------------------------------

## What is RAG?

**RAG = Retrieval-Augmented Generation.**

RAG combines:

1.  **Retrieval** --- search your own documents for relevant
    information.
2.  **Generation** --- give the retrieved information to an LLM and
    generate a natural-language answer.

The basic flow is:

``` text
User Question
      ↓
Question Embedding
      ↓
Vector Search
      ↓
Relevant Document Chunks
      ↓
Context + Question
      ↓
LLM
      ↓
Answer + Sources
```

### Simple Example

Suppose a PDF says:

``` text
SYMBIO is a Living-City Operating System designed to
connect isolated municipal departments.
```

The user asks:

``` text
What is SYMBIO?
```

RAG searches the document for semantically relevant text, retrieves the
chunk containing the definition, and supplies it to the LLM.

The LLM then produces an answer grounded in that retrieved context.

------------------------------------------------------------------------

## Why RAG?

Sending an entire large document to an LLM for every question can be
inefficient.

RAG instead searches the document first and sends only relevant context.

  -----------------------------------------------------------------------
  Traditional approach                RAG
  ----------------------------------- -----------------------------------
  Large context may be sent           Relevant chunks are retrieved
  repeatedly                          

  Can become expensive for large      Reduces unnecessary context
  documents                           

  Document knowledge is not indexed   Document chunks are indexed
  locally                             

  Harder to trace source text         Chunks can retain page metadata
  -----------------------------------------------------------------------

RAG does not guarantee perfect answers. Retrieval quality, chunking,
embeddings, prompts, document extraction, and the LLM all affect the
final result.

------------------------------------------------------------------------

## Features

-   📄 PDF upload
-   🔍 Semantic search
-   🧠 Sentence Transformer embeddings
-   ⚡ FAISS vector similarity search
-   🤖 LLM-based question answering
-   📚 Source-page references
-   💻 Streamlit interface
-   💾 Existing vector-store loading
-   ☁️ Streamlit Community Cloud deployment

------------------------------------------------------------------------

## Technology Stack

  Technology                  Purpose
  --------------------------- -----------------------------------------
  Python                      Core language
  Streamlit                   Web interface
  pypdf                       PDF text extraction
  Sentence Transformers       Text embeddings
  FAISS                       Vector similarity search
  NumPy                       Numerical/vector operations
  python-dotenv               Environment variables
  Google GenAI                Gemini/Google generative AI integration
  Git/GitHub                  Version control
  Streamlit Community Cloud   Deployment

------------------------------------------------------------------------

## How It Works

DocuChat has two major pipelines.

### 1. Document Ingestion

``` text
PDF Upload
    ↓
pypdf
    ↓
Extract Text
    ↓
Split Into Chunks
    ↓
Sentence Transformer
    ↓
Embeddings
    ↓
FAISS Index
    ↓
Vector Store
```

### 2. Question Answering

``` text
User Question
    ↓
Question Embedding
    ↓
FAISS Similarity Search
    ↓
Top Relevant Chunks
    ↓
Context + Prompt
    ↓
Gemini / LLM
    ↓
Answer
    ↓
Source Pages
```

------------------------------------------------------------------------

## Architecture

``` mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]

    UI -->|Upload PDF| ING[Document Ingestion]
    ING --> PDF[pypdf PDF Reader]
    PDF --> TXT[Extract Text]
    TXT --> CHUNK[Text Chunking]
    CHUNK --> EMB[Sentence Transformer]
    EMB --> VDB[(FAISS Vector Index)]

    UI -->|Ask Question| Q[User Query]
    Q --> QEMB[Query Embedding]
    QEMB --> SEARCH[FAISS Similarity Search]
    VDB --> SEARCH

    SEARCH --> RET[Relevant Document Chunks]
    RET --> PROMPT[Prompt + Retrieved Context]
    PROMPT --> LLM[Gemini / LLM]
    LLM --> ANSWER[Generated Answer]
    RET --> SOURCE[Source Page Metadata]

    ANSWER --> UI
    SOURCE --> UI
```

### Component Responsibilities

#### `app.py`

The Streamlit entrypoint and user interface.

Typical responsibilities:

-   PDF upload
-   Question input
-   Displaying answers
-   Displaying sources
-   Calling RAG functions

#### `rag.py`

The RAG engine.

The current project imports RAG functionality such as:

``` python
load_embedding_model
load_vector_store
process_document
```

The module is responsible for the document-processing, embedding,
vector-store, retrieval, and generation logic used by the application.

#### `requirements.txt`

Defines the Python packages needed to run and deploy the project.

------------------------------------------------------------------------

## Project Structure

A typical project structure is:

``` text
docuchat-rag/
│
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── data/
│   └── documents/
│
└── vector_store/
    └── ...
```

The exact vector-store files/directories depend on the implementation.

------------------------------------------------------------------------

# Build From Scratch

If you are new to RAG, build it in this order.

## Step 1 --- Create the project

``` bash
mkdir docuchat-rag
cd docuchat-rag
```

## Step 2 --- Create a Python virtual environment

Python **3.11** is recommended for this project because it provides a
stable compatibility baseline for the ML/RAG dependencies and matches
the current Streamlit deployment configuration.

### macOS / Linux

``` bash
python3.11 -m venv venv
source venv/bin/activate
```

### Windows

``` powershell
py -3.11 -m venv venv
venv\Scripts\activate
```

Verify:

``` bash
python --version
```

Expected:

``` text
Python 3.11.x
```

## Step 3 --- Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Current dependencies:

``` text
streamlit
pypdf
sentence-transformers
faiss-cpu
numpy
python-dotenv
google-genai
```

Verify important packages:

``` bash
python -c "import streamlit, pypdf, faiss, numpy; print('Dependencies OK')"
```

------------------------------------------------------------------------

# Installation

## Environment variables

If the application uses a Gemini API key, create:

``` text
.env
```

Example:

``` env
GEMINI_API_KEY=your_api_key_here
```

Use the exact variable name expected by your code.

Never commit real credentials.

Recommended `.gitignore`:

``` gitignore
.env
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

------------------------------------------------------------------------

# Run Locally

Start the application:

``` bash
streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

### Test

1.  Upload a PDF.
2.  Wait until the document is processed.
3.  Ask a question.
4.  Check the generated answer.
5.  Check the displayed source pages.

------------------------------------------------------------------------

# Understanding the RAG Pipeline

## 1. PDF Extraction

`pypdf` reads PDF pages and extracts their text.

Conceptually:

``` python
from pypdf import PdfReader

reader = PdfReader(pdf_file)

for page in reader.pages:
    text = page.extract_text()
```

## 2. Chunking

A large document is split into smaller pieces:

``` text
Full Document
    ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

Chunks make semantic retrieval practical and allow page metadata to be
associated with individual pieces of text.

## 3. Embeddings

An embedding represents text as a numerical vector.

Conceptually:

``` text
"Explain machine learning"
        ↓
[0.12, -0.43, 0.88, ...]
```

Semantically similar sentences tend to have similar vectors.

## 4. FAISS

FAISS searches vectors efficiently.

``` text
Document Chunks
      ↓
Embeddings
      ↓
FAISS Index
```

At query time:

``` text
Question
   ↓
Question Embedding
   ↓
FAISS Search
   ↓
Most Relevant Chunks
```

## 5. Prompt Construction

The retrieved chunks become context for the LLM.

Conceptually:

``` text
Document Context:
[retrieved chunk 1]
[retrieved chunk 2]
[retrieved chunk 3]

Question:
What is the main idea?

        ↓

Gemini / LLM

        ↓

Answer
```

## 6. Sources

Retrieved chunks can retain metadata such as page numbers:

``` text
Chunk
 ├── text
 └── page = 11
```

This allows the application to show source references such as:

``` text
Page 11
Page 12
Page 3
```

------------------------------------------------------------------------

# One Diagram to Remember

``` text
                 YOUR PDF
                    │
                    ▼
              Extract Text
                    │
                    ▼
                 Chunks
                    │
                    ▼
               Embeddings
                    │
                    ▼
              FAISS Index
                    │
                    │
USER ──────► QUESTION
                    │
                    ▼
            Query Embedding
                    │
                    ▼
             Vector Search
                    │
                    ▼
          Relevant PDF Chunks
                    │
                    ▼
              Context + Q
                    │
                    ▼
                 LLM
                    │
                    ▼
              Final Answer
                    │
                    ▼
             Source Pages
```

------------------------------------------------------------------------

# Vector Database Explained

A vector store is a system that stores embeddings and allows similarity
searches.

Example:

``` text
Chunk A → [0.1, 0.2, 0.8, ...]
Chunk B → [0.7, 0.1, 0.3, ...]
Chunk C → [0.2, 0.9, 0.4, ...]
```

A query is also converted to a vector:

``` text
Question → [0.12, 0.21, 0.79, ...]
```

The vector index searches for nearby vectors.

That is the core mechanism behind semantic retrieval.

------------------------------------------------------------------------

# Deployment

This project can be deployed using Streamlit Community Cloud.

## GitHub

Your repository should contain at least:

``` text
app.py
rag.py
requirements.txt
README.md
```

Push changes:

``` bash
git add .
git commit -m "Update DocuChat RAG"
git push origin main
```

## Streamlit

Create/deploy the application from the GitHub repository.

Use:

``` text
Repository: YOUR_USERNAME/YOUR_REPOSITORY
Branch: main
Main file: app.py
```

### Important

Use:

``` text
app.py
```

as the Streamlit entrypoint.

Do **not** use:

``` text
rag.py
```

as the entrypoint because `rag.py` is the RAG module imported by the
Streamlit application.

## Secrets

Do not commit `.env` with real API keys.

Configure the required secret in Streamlit's Secrets settings using the
same variable name expected by your code.

------------------------------------------------------------------------

# Troubleshooting

## `ModuleNotFoundError: No module named 'pypdf'`

Make sure the virtual environment is active:

``` bash
source venv/bin/activate
```

Install with the active interpreter:

``` bash
python -m pip install pypdf
```

Verify:

``` bash
python -c "import pypdf; print(pypdf.__version__)"
```

If your shell has a `python` alias, check:

``` bash
type -a python
```

You can bypass the alias:

``` bash
./venv/bin/python -m pip install pypdf
```

## Python version mismatch

Check:

``` bash
python --version
```

and:

``` bash
python -m pip --version
```

Both should point to the same virtual environment.

Using Python 3.11 for local development and deployment provides a
consistent baseline.

## FAISS installation error

Try:

``` bash
python -m pip install --upgrade pip
python -m pip install faiss-cpu
```

Verify:

``` bash
python -c "import faiss; print('FAISS OK')"
```

## Streamlit page is blank

Check the terminal for:

``` text
Traceback
ModuleNotFoundError
API errors
```

Run:

``` bash
streamlit run app.py
```

not:

``` bash
streamlit run rag.py
```

## Deployment works locally but fails in Streamlit Cloud

Check:

1.  `requirements.txt` is committed.
2.  `app.py` is the entrypoint.
3.  Required secrets exist.
4.  GitHub contains the latest commit.
5.  The deployment Python version is compatible.
6.  Deployment logs for import or API errors.

------------------------------------------------------------------------

# Security

Never commit:

``` text
API keys
Passwords
Tokens
Service-account credentials
.env
.streamlit/secrets.toml
```

If a secret is accidentally pushed to GitHub, revoke and rotate it
immediately.

------------------------------------------------------------------------

# Future Improvements

Possible upgrades include:

### Better Retrieval

-   Recursive chunking
-   Chunk overlap
-   Hybrid keyword + vector search
-   Reranking
-   Metadata filtering
-   Query rewriting
-   Multi-query retrieval

### More Documents

-   DOCX
-   TXT
-   Markdown
-   CSV
-   Web pages
-   Multiple PDFs

### AI Features

-   Document summaries
-   Multi-document chat
-   Conversation memory
-   Follow-up questions
-   Citation verification
-   Table extraction
-   Structured answers

### Production Infrastructure

A larger deployment could evolve from local FAISS toward a dedicated
vector database:

``` text
Streamlit / Frontend
        ↓
API Backend
        ↓
Document Processing
        ↓
Embedding Model
        ↓
Vector Database
        ↓
LLM
```

Possible vector-database choices include Qdrant, Milvus, Weaviate, or
Pinecone.

------------------------------------------------------------------------

# Learning Roadmap

If you know nothing about RAG, learn in this order:

## Level 1 --- Python

Learn:

-   Functions
-   Classes
-   Modules
-   File handling
-   Exceptions
-   Virtual environments
-   pip

## Level 2 --- Machine Learning Basics

Understand:

-   Vectors
-   Similarity
-   Cosine similarity
-   Neural networks
-   Transformers

## Level 3 --- NLP

Learn:

-   Tokens
-   Embeddings
-   Semantic similarity
-   Transformers
-   Context

## Level 4 --- Vector Search

Understand:

``` text
Text
 ↓
Embedding
 ↓
Vector
 ↓
Index
 ↓
Similarity Search
```

## Level 5 --- RAG

Understand:

``` text
Ingestion
 ↓
Chunking
 ↓
Embedding
 ↓
Retrieval
 ↓
Prompt Construction
 ↓
Generation
 ↓
Citations
```

## Level 6 --- Production RAG

Learn:

-   Retrieval evaluation
-   Reranking
-   Hallucination reduction
-   Caching
-   Observability
-   Authentication
-   Rate limiting
-   Cost optimization

------------------------------------------------------------------------

# RAG in One Sentence

> **RAG retrieves relevant information from your own data and supplies
> that information to an LLM so the model can generate a
> context-grounded answer.**

------------------------------------------------------------------------

## ⭐ Contributing

``` bash
git clone YOUR_REPOSITORY_URL
cd docuchat-rag

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Create a feature branch:

``` bash
git checkout -b feature/your-feature
```

Test locally, commit your changes, and open a pull request.

------------------------------------------------------------------------

## License

Add your preferred license to the repository, for example:

``` text
MIT License
```

If using MIT, add a corresponding `LICENSE` file.

------------------------------------------------------------------------

## ⭐ Final Mental Model

``` text
PDF
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieve
 ↓
Context
 ↓
Gemini / LLM
 ↓
Answer + Sources
```

**That's DocuChat AI. That's RAG.**
