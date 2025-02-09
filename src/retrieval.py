from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.INFO)

def create_retriever(vector_db, llm=None, resume = False, job_desc = False, answer = False):

    QUERY_PROMPT = PromptTemplate(
        input_variables = ["question"],
        template = """You are an AI assistant that helps with resumes. Your job is to generate seven different versions of the
        user's question based to retrieve the most relevant documents from a vector database. By generating multiple perpspectives
        of the user's question, your goal is to find the most relevant documents in the vector database. Provide these 
        alternate questions separated by new lines.
        Original Question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    logging.info(f"Created retriever")
    return retriever

def create_chain(resume_retriever=None, job_retriever=None, llm=None, resume=False, job=False):

    template = """ Answer based on the following context of resume and job context. You can choose to consider one 
    context over the other or consider both depending if the context is relevant to the user's question. 
     if job context is empty answer based on resume context:
        Context: {context}
        Question: {question}
    """

    both_template = """Answer the user question based on the following context of resume and job context.
        Resume Context: {resume_context}
        Job Context: {job_context}
        Question: {question}
    """

    single_prompt = ChatPromptTemplate.from_template(template)
    both_prompt = ChatPromptTemplate.from_template(both_template)

    if resume and job:
        chain = (
            {"resume_context": resume_retriever, "job_context": job_retriever, "question": RunnablePassthrough()}
            | both_prompt
            | llm
            | StrOutputParser()
        )
    elif resume:
        chain = (
            {"context": resume_retriever, "question": RunnablePassthrough()}
            | single_prompt
            | llm
            | StrOutputParser()
        )
    elif job:
        chain = (
            {"context": job_retriever, "question": RunnablePassthrough()}
            | single_prompt
            | llm
            | StrOutputParser()
        )
    else:
        print("error")

    logging.info("Created chain with preserved syntax")
    return chain

