## Resume-Bot
This is an implementation of a Retrieval Augmented Generation (RAG) pipeline for the Qwen-2.5 model to anlyze job descriptions in context of job descriptions. The RAG pipeline creates a vector database of the uploaded files and responds with relevant context it retrieves from the knowledge base. 

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