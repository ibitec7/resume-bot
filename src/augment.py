from ingestion import *
from augment import *
from retrieval import *
import streamlit as st
from langchain_ollama import ChatOllama

MODEL_NAME = "qwen2.5:3b"
MODEL_FILE = "./Modelfile"

def main():
    st.title("Resume Analyzer")

    uploaded_resume = st.file_uploader("Upload your resume here", type=["pdf","docx","txt"])
    uploaded_job_desc = st.file_uploader("Upload the job description here", type=["pdf","docx","txt"])

    if uploaded_resume is not None:
        temp_dir = "./tmp"
        os.makedirs(temp_dir, exist_ok=True)
        resume_path = os.path.join(temp_dir, uploaded_resume.name)         

        with open(resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())

        st.success(f"File {uploaded_resume.name} uploaded successfully.")
        resume_db, _ = load_vector_db(doc_path=resume_path, persist_dir="./resume_db", name="resume")

    if uploaded_job_desc is not None:
            job_desc_path = os.path.join(temp_dir, uploaded_job_desc.name)
            
            with open(job_desc_path, "wb") as f:
                f.write(uploaded_job_desc.getbuffer())
            
            st.success(f"File {uploaded_job_desc.name} uploaded successfully.")

            job_desc_db, _ = load_vector_db(doc_path=job_desc_path, persist_dir="./job_desc_db", name="job_desc")

    else:
        print("error")

    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

                if resume_db is None or job_desc_db is None:
                    st.error("Failed to load vector store")
                    return
                
                if resume_db is not None:
                    resume_retriever = create_retriever(resume_db, llm)

                if job_desc_db is not None:
                    job_retriever = create_retriever(job_desc_db, llm)

                
                if job_desc_db is not None and resume_db is not None:
                    context = ollama.chat(model=MODEL_NAME,messages=[
                        {"role": "system", "content": f"Tell me if the following question requires\
                             the context of the job description, the resume, or both.\
                                 Answer by saying one word that is 'job', 'resume', or 'both' in lowercase.\
                                     Question: {user_input}"},
                    ])
                    context = context.message.content.lower()
                    st.write(context)

                elif job_desc_db is not None:
                    context = "job"

                elif resume_db is not None:
                    context = "resume"

                if context == "resume":
                    chain = create_chain(resume_retriever=resume_retriever, llm=llm, resume=True)
                elif context == "job":
                    chain = create_chain(job_retriever=job_retriever, llm=llm, job=True)
                elif context == "both":
                    chain = create_chain(resume_retriever=resume_retriever, job_retriever=job_retriever, llm=llm, resume=True, job=True)

                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                return
            
    else:
        st.info("Please enter a question to get started")

if __name__ == "__main__":
    main()