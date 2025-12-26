from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from src.prompts import BASE_PROMPT

def run_langchain_rag(chunks, query):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant policy information was found.", []

    prompt = PromptTemplate(
        template=BASE_PROMPT,
        input_variables=["context", "question"]
    )

    llm = OllamaLLM(
        model="mistral",
        temperature=0
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # 6. Retrieval chain (modern replacement for RetrievalQA)
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    # 7. Invoke
    response = retrieval_chain.invoke({"input": query})

    sources = sorted({
        doc.metadata.get("source", "unknown")
        for doc in response["context"]
    })

    return response["answer"], sources
