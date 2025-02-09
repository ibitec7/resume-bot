from ingestion import *
import unittest
import logging
import os
from retrieval import *

logging.basicConfig(level=logging.INFO)

class TestIngestion(unittest.TestCase):
    def test_creation(self):
        vector_db, chunk_ids = load_vector_db(doc_path="./test.pdf")
        logging.info(f"Loaded vector store with {len(chunk_ids)} documents")
        exists = os.path.exists("./chroma_db")
        self.assertGreater(len(chunk_ids), 1)
        self.assertTrue(exists)

    def test_addition_removal(self):
        docs = ingest_document(doc_path="./test_add.pdf")
        chunks, chunk_ids = split_documents(docs)
        vector_db, _ = load_vector_db(doc_path="./test.pdf")
        initial_len = len(vector_db._collection.get()['ids'])

        add_documents(vector_db, chunks, chunk_ids)
        self.assertGreater(len(vector_db._collection.get()['ids']), initial_len)

        remove_documents(vector_db, chunk_ids)
        self.assertEqual(len(vector_db._collection.get()['ids']), initial_len)

class TestRetrieval(unittest.TestCase):
    def test_retrieval(self):
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.7)
        vector_db, chunk_ids = load_vector_db(doc_path="./test.pdf", llm=llm)

        logging.info(f"Loaded vector store with {len(chunk_ids)} documents")
        exists = os.path.exists("./chroma_db")
        self.assertGreater(len(chunk_ids), 1)
        self.assertTrue(exists)

        docs = ingest_document(doc_path="./test_add.pdf")
        chunks, chunk_ids = split_documents(docs)
        vector_db, _ = load_vector_db(doc_path="./test.pdf")
        initial_len = len(vector_db._collection.get()['ids'])

        add_documents(vector_db, chunks, chunk_ids)
        self.assertGreater(len(vector_db._collection.get()['ids']), initial_len)

        job_desc_db, _ = load_vector_db(doc_path="./test_job_desc.pdf")
        resume_db, _ = load_vector_db(doc_path="./test_resume.pdf")

        job_retriever = create_retriever(job_desc_db)
        resume_retriever = create_retriever(resume_db)

        self.assertIsNotNone(job_retriever)
        self.assertIsNotNone(resume_retriever)

        job_retriever = create_retriever(job_desc_db, llm)
        resume_retriever = create_retriever(resume_db, llm)

        self.assertIsNotNone(job_retriever)
        self.assertIsNotNone(resume_retriever)

        remove_documents(vector_db, chunk_ids)
        self.assertEqual(len(vector_db._collection.get()['ids']), initial_len)

if __name__ == "__main__":
    unittest.main()
