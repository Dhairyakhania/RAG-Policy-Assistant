from src.load_docs import load_documents
from src.chunking import chunk_documents
# from src.baseline_rag import run_baseline_rag
from src.langchain_rag import run_langchain_rag

def main():
    docs = load_documents()
    chunks = chunk_documents(docs)

    print("Documents loaded and chunked.")

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        # print("\n--- Baseline RAG ---")
        # print(run_baseline_rag(chunks, query))

        print("\n--- LangChain RAG ---")
        answer, sources = run_langchain_rag(chunks, query)
        print(answer)

        if sources:
            print("\nSources:")
            for s in sources:
                print("-", s)

if __name__ == "__main__":
    main()
