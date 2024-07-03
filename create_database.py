import os
import shutil

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    example = chunks[2]
    print(f"Example chunk: {example.page_content}")
    print(example.metadata)

    return chunks


def save_to_chroma(chunks) -> None:
    # Clear database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create new database
    Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        collection_name="books",
        persist_directory=CHROMA_PATH,
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()
