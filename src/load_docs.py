from langchain_community.document_loaders import TextLoader
from pathlib import Path


def load_documents(data_dir="data"):
    docs = []

    for file in Path(data_dir).glob("*.txt"):
        loader = TextLoader(
            str(file),
            encoding="utf-8",
            autodetect_encoding=True
        )
        docs.extend(loader.load())

    return docs
