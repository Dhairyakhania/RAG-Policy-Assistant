from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from src.prompts import BASE_PROMPT
import json

CONFIDENCE_THRESHOLD = 0.25
REFUSAL_TEXT = "The provided documents do not contain this information."


def run_langchain_rag(chunks, query):
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Vector store & retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Retrieve documents (modern API)
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return REFUSAL_TEXT

    # 4. Confidence-based retrieval gating
    query_embedding = embeddings.embed_query(query)
    similarities = [
        sum(q * d for q, d in zip(
            query_embedding,
            embeddings.embed_documents([doc.page_content])[0]
        ))
        for doc in retrieved_docs
    ]

    max_similarity = max(similarities)
    if max_similarity < CONFIDENCE_THRESHOLD:
        return REFUSAL_TEXT

    # 5. Prompt
    prompt = PromptTemplate(
        template=BASE_PROMPT,
        input_variables=["context", "input"]
    )

    # 6. Local LLM
    llm = OllamaLLM(model="mistral", temperature=0)

    # 7. Document QA chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # 8. Retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # 9. Generate answer
    response = retrieval_chain.invoke({"input": query})
    output = response["answer"].strip()

    # 10. Hard refusal enforcement
    if output == REFUSAL_TEXT:
        return REFUSAL_TEXT

    # 11. Enforce JSON structure
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return REFUSAL_TEXT

    # 12. Reject weak / yes-no answers
    answer_text = parsed.get("answer", "").strip().lower()
    if answer_text in {"yes", "no", "yes.", "no."} or len(answer_text.split()) < 5:
        return REFUSAL_TEXT

    # 13. Source alignment
    sources = {
        doc.metadata.get("source", "unknown")
        for doc in response["context"]
    }
    parsed["source_documents"] = sorted(list(sources))

    # 14. Confidence score
    parsed["confidence"] = round(min(1.0, max_similarity), 2)

    return parsed
