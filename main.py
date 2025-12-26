from src.load_docs import load_documents
from src.chunking import chunk_documents
from src.langchain_rag import run_langchain_rag


def format_user_answer(result):
    if isinstance(result, str):
        return result

    answer = result["answer"]
    sources = ", ".join(result["source_documents"])
    confidence = int(result["confidence"] * 100)

    return (
        f"{answer}\n\n"
        f"Sources: {sources}\n"
        f"Confidence: {confidence}%"
    )


def main():
    docs = load_documents()
    chunks = chunk_documents(docs)

    print("Policy documents loaded and indexed.")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        result = run_langchain_rag(chunks, query)
        print("\n" + format_user_answer(result))


if __name__ == "__main__":
    main()
