import streamlit as st

from src.load_docs import load_documents
from src.chunking import chunk_documents
from src.langchain_rag import run_langchain_rag


@st.cache_resource
def load_rag_pipeline():
    docs = load_documents()
    chunks = chunk_documents(docs)
    return chunks


def format_answer(result):
    if isinstance(result, str):
        return result, None, None

    return (
        result["answer"],
        result["source_documents"],
        int(result["confidence"] * 100),
    )


# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Policy RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Policy Question-Answering Assistant")
st.caption("Retrieval-Augmented Generation over company policy documents")

st.markdown(
    """
Ask questions related to returns, refunds, cancellations, delivery options,
or customer service policies.  
The assistant answers **only from the documents** and refuses unsupported queries.
"""
)

chunks = load_rag_pipeline()

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. Can I cancel an order after it has shipped?"
)

if query:
    with st.spinner("Searching policy documents..."):
        result = run_langchain_rag(chunks, query)

    answer, sources, confidence = format_answer(result)

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources")
        for s in sources:
            st.write(f"- {s}")

    if confidence is not None:
        st.subheader("Confidence")
        st.progress(confidence / 100)
        st.caption(f"{confidence}%")
