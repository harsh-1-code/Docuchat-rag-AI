import streamlit as st
import os

from rag import (
    load_embedding_model,
    load_vector_store,
    process_document,
    ask_question
)


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================
# SESSION STATE
# ============================================

if "model" not in st.session_state:

    with st.spinner(
        "Loading AI embedding model..."
    ):

        st.session_state.model = (
            load_embedding_model()
        )


if "index" not in st.session_state:

    index, chunks = load_vector_store()

    st.session_state.index = index
    st.session_state.chunks = chunks


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.markdown(
        "## 📄 Document"
    )

    st.markdown(
        "### Upload a PDF"
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(
            f"Selected: {uploaded_file.name}"
        )

        process_button = st.button(
            "🚀 Process Document",
            use_container_width=True
        )

        if process_button:

            os.makedirs(
                "data",
                exist_ok=True
            )

            pdf_path = (
                "data/current_document.pdf"
            )

            with open(
                pdf_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            with st.spinner(
                "Processing document..."
            ):

                stats = process_document(
                    pdf_path,
                    st.session_state.model
                )

                index, chunks = load_vector_store()

                st.session_state.index = index
                st.session_state.chunks = chunks

                st.session_state.messages = []

            st.success(
                "Document processed successfully!"
            )

            st.write(
                f"📄 Pages: {stats['pages']}"
            )

            st.write(
                f"🧩 Chunks: {stats['chunks']}"
            )

            st.write(
                f"🔢 Vectors: {stats['vectors']}"
            )


# ============================================
# MAIN HEADER
# ============================================

st.markdown(
    '<div class="main-title">📚 DocuChat AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Chat with your documents using RAG'
    '</div>',
    unsafe_allow_html=True
)


# ============================================
# DOCUMENT STATUS
# ============================================

if st.session_state.index is None:

    st.info(
        "👉 Upload a PDF and click "
        "**Process Document** to begin."
    )

else:

    st.success(
        f"Document ready • "
        f"{st.session_state.index.ntotal} vectors"
    )


# ============================================
# CHAT HISTORY
# ============================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================
# CHAT INPUT
# ============================================

question = st.chat_input(
    "Ask something about your document..."
)


if question:

    # ----------------------------------------
    # User message
    # ----------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # ----------------------------------------
    # Check document
    # ----------------------------------------

    if st.session_state.index is None:

        answer = (
            "Please upload and process a PDF first."
        )

        sources = []

    else:

        # ------------------------------------
        # RAG
        # ------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching document..."
            ):

                answer, sources = ask_question(
                    question,
                    st.session_state.model,
                    st.session_state.index,
                    st.session_state.chunks,
                    k=5
                )

            st.markdown(
                answer
            )

            # --------------------------------
            # Sources
            # --------------------------------

            if sources:

                st.markdown(
                    "### 📚 Sources"
                )

                seen_pages = set()

                for source in sources:

                    page = source["page"]

                    if page not in seen_pages:

                        st.caption(
                            f"📄 Page {page}"
                        )

                        seen_pages.add(page)

    # ----------------------------------------
    # Save assistant message
    # ----------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })