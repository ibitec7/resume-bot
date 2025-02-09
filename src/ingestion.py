from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os
import logging
import uuid
import ollama
import streamlit as st

## My environment variables
DOC_PATH = "/home/ibrahim/Downloads/Resume-1.pdf"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "knowledge"
MODEL_NAME = "qwen2.5:3b"
PERSIST_DIR = "./chroma_db"

logging.basicConfig(level=logging.INFO)

# My function to ingest the documents 
def ingest_document(doc_path):
    if os.path.exists(doc_path):
        if doc_path.endswith(".pdf"):
            loader = UnstructuredPDFLoader(file_path=doc_path)
        elif doc_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path=doc_path)
        elif doc_path.endswith(".txt"):
            loader = TextLoader(file_path=doc_path)
        else:
            logging.error(f"Unsupported file format: {doc_path}")
            return None

        data = loader.load()
        logging.info(f"Loaded resume")
        return data

    else:
        logging.error(f"File does not exist: {doc_path}")
        return None

# My function to split the documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        uid = uuid.uuid4()
        chunk_ids.append(f"{uid}")
        chunk.metadata["chunk_id"] = f"{uid}"
    logging.info(f"Split documents into {len(chunks)} chunks")
    return chunks, chunk_ids

# Here I embed the documents into vectors and load them into a vector store
def load_vector_db(doc_path=DOC_PATH, persist_dir="./chroma_db", name="knowledge"):
    # Load the embedding model and wrap it around the OllamaEmbeddings class
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Check if the vector store already exists
    if os.path.exists(persist_dir) and os.listdir(persist_dir) and persist_dir == "./chroma_db":
        # If it does, load the existing vector store
        vector_db = Chroma(
            collection_name=name,
            embedding_function=embedding,
            persist_directory=persist_dir,
        )
        logging.info(f"Loading existing vector store")
        chunk_ids = vector_db._collection.get()['ids']
    
    elif os.path.exists(persist_dir) and os.listdir(persist_dir) and persist_dir != "./chroma_db":
        vector_db = Chroma(
            collection_name=name,
            embedding_function=embedding,
            persist_directory=persist_dir,
        )

        logging.info(f"Loaded existing vector store")
        chunk_ids = vector_db._collection.get()['ids']
        remove_documents(vector_db, chunk_ids)
        logging.info(f"Cleared documents from vector store")

        data = ingest_document(doc_path)
        if data is None:
            return None, None

        chunks, chunk_ids = split_documents(data)

        add_documents(vector_db, chunks, chunk_ids)
        logging.info(f"Added new documents to vector store")

        return vector_db, chunk_ids

    else:
        # If it does not exist, create a new vector store
        data = ingest_document(doc_path)
        if data is None:
            return None, None

        chunks, _ = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=name,
            persist_directory=persist_dir,
        )

        logging.info(f"Created new vector store")

        chunk_ids = vector_db._collection.get()['ids']

    return vector_db, chunk_ids

# My function to add documents to the Chromadb
def add_documents(vector_db, docs, doc_ids):
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    embedded_docs = embedding.embed_documents([doc.page_content for doc in docs])

    vector_db._collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
        embeddings=embedded_docs,
        ids=doc_ids
    )

# A function to remove documents by id from the Chromadb
def remove_documents(vector_db, doc_ids):
    vector_db._collection.delete(ids=doc_ids)

def main():
    vector_db, chunk_ids = load_vector_db(doc_path="/home/ibrahim/Downloads/Resume-1.pdf")
    logging.info(f"Loaded vector store with {len(chunk_ids)} documents")

    docs = vector_db._collection.get()

    print(help(vector_db._collection.add))
    # remove_documents(vector_db, chunk_ids)
    # logging.info(f"Removed documents from vector store")

    # remaining_ids = vector_db._collection.get()['ids']
    # for chunk_id in chunk_ids:
    #     if chunk_id in remaining_ids:
    #         logging.error(f"Chunk ID {chunk_id} was not removed")
    #     else:
    #         logging.info(f"Chunk ID {chunk_id} was successfully removed")

if __name__ == "__main__":
    main()