## Resume-Bot
This is an implementation of a Retrieval Augmented Generation (RAG) pipeline for the Qwen-2.5 model to analyze resumes in context of job descriptions. The RAG pipeline creates a vector database of the uploaded files and responds with relevant context it retrieves from the knowledge base.

## Directories
- **src/:** contains the source code for the RAG.
- **src/ingest.py:** script to ingest documents, create a vector database and add/remove documents.
- **src/retrieval.py:** script to for the LLM to retrieve the relevant context from the knowledge base.
- **src/augment.py:** script to augment the user prompt for the LLM and the UI for user interaction.
- **src/scrapper.py** script for retrieval of HTML scripts from URLs to enable web-searches (work in progress).
- **src/test.py:** script for unit testing of the pipeline.
- **src/test*.pdf:** pdf files to test the ingestions of the database.
- **requirements.txt:** text file for the  requirements of the project.

## Requirements
1. **GPU:** to run the model locally
2. **requirements.txt:** this lists all the dependencies such as Ollama, LangChain, Pytorch, TensorRT, etc.
3. **Python:** version >= 3.12.7

## How to run
1. Create a python virtual environment
```
    python -m venv venv
```

2. Activate the virtual environment:
```
    source venv/bin/activate
```

3. Install the dependencies:
```
    pip install -r requirements.txt
```

4. Run the UI application:
```
    streamlit run src/augment.py
```

5. Interact and use the UI with features to upload and clear resumes and job description contexts
