from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================
# ENVIRONMENT
# ============================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================
# EMBEDDING MODEL
# ============================================

def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================
# PDF TEXT EXTRACTION
# ============================================

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# ============================================
# CHUNKING
# ============================================

def create_chunks(
    pages,
    chunk_size=1500,
    overlap=300
):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# ============================================
# CREATE FAISS VECTOR STORE
# ============================================

def create_vector_store(
    chunks,
    model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================
# SAVE VECTOR STORE
# ============================================

def save_vector_store(
    index,
    chunks,
    index_path="storage/index.faiss",
    chunks_path="storage/chunks.json"
):

    os.makedirs(
        "storage",
        exist_ok=True
    )

    # Save FAISS
    faiss.write_index(
        index,
        index_path
    )

    # Save chunks
    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================
# LOAD VECTOR STORE
# ============================================

def load_vector_store(
    index_path="storage/index.faiss",
    chunks_path="storage/chunks.json"
):

    if not os.path.exists(index_path):
        return None, None

    if not os.path.exists(chunks_path):
        return None, None

    index = faiss.read_index(
        index_path
    )

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks


# ============================================
# PROCESS DOCUMENT
# ============================================

def process_document(
    pdf_path,
    model
):

    print("Extracting PDF text...")

    pages = extract_text_from_pdf(
        pdf_path
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    print("Creating chunks...")

    chunks = create_chunks(
        pages
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print("Creating embeddings...")

    index = create_vector_store(
        chunks,
        model
    )

    print(
        f"Vectors created: {index.ntotal}"
    )

    save_vector_store(
        index,
        chunks
    )

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "vectors": index.ntotal
    }


# ============================================
# SEARCH
# ============================================

def search_chunks(
    question,
    model,
    index,
    chunks,
    k=5
):

    query_embedding = model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        min(k, index.ntotal)
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(distance)
        })

    return results


# ============================================
# GENERATE ANSWER
# ============================================

def generate_answer(
    question,
    results
):

    context = "\n\n".join(
        [
            f"Page {result['page']}:\n"
            f"{result['text']}"
            for result in results
        ]
    )

    prompt = f"""
You are DocuChat AI, a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not present in the context, say:
   "I couldn't find this information in the document."
4. Give a clear and concise answer.
5. When useful, mention the relevant page number.
6. If multiple parts of the document answer the question,
   combine them into one coherent answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ============================================
# COMPLETE QUERY PIPELINE
# ============================================

def ask_question(
    question,
    model,
    index,
    chunks,
    k=5
):

    results = search_chunks(
        question,
        model,
        index,
        chunks,
        k=k
    )

    answer = generate_answer(
        question,
        results
    )

    return answer, results


# ============================================
# TERMINAL TEST
# ============================================

if __name__ == "__main__":

    PDF_PATH = (
        "data/Gemini AI futuristic SYMBIOS city.pdf"
    )

    model = load_embedding_model()

    # Check existing vector store
    index, chunks = load_vector_store()

    if index is None:

        print(
            "No saved vector store found."
        )

        print(
            "Processing PDF..."
        )

        process_document(
            PDF_PATH,
            model
        )

        index, chunks = load_vector_store()

    else:

        print(
            "Loaded existing vector store!"
        )

        print(
            f"Vectors: {index.ntotal}"
        )

    # Ask questions continuously
    while True:

        question = input(
            "\nAsk something about the PDF "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        answer, results = ask_question(
            question,
            model,
            index,
            chunks
        )

        print(
            "\n=============================="
        )

        print(
            "🤖 ANSWER"
        )

        print(
            "==============================\n"
        )

        print(answer)

        print(
            "\n=============================="
        )

        print(
            "📚 SOURCES"
        )

        print(
            "==============================\n"
        )

        seen_pages = set()

        for result in results:

            page = result["page"]

            if page not in seen_pages:

                print(
                    f"Page {page}"
                )

                seen_pages.add(page)