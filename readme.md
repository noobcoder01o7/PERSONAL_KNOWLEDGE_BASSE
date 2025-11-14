# AI-Powered Personal Knowledge Base
# PRERIQUISITES
This project is a personal knowledge base that uses a Large Language Model (LLM) to answer questions based on your own document

Before you begin, you must install **Ollama** and download the Llama 3 model.

1.  Download and install Ollama from [https://ollama.com/](https://ollama.com/).
2.  In your terminal, run the following command to pull the Llama 3 model:
    ```bash
    ollama pull llama3
    ```
3.  Ensure the Ollama application is running in the background.

#SETUP INSTRUCTIONS

1.  Clone the repository or download and unzip the project folder.

2.  Create a virtual environment :
    '''
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    '''

3.  Install the required libraries**:
    '''
    pip install -r requirements.txt
    '''

## How to Run

1.  Add Your Documents: Place your text-based '.pdf' or '.txt' files into the 'documents' folder. A sample file is already included.

2.  Launch Jupyter: Open your terminal in the project folder and run:
    '''
    jupyter notebook
    '''

3.  Run the Notebook: Open the 'project.ipynb' file and run the cells from top to bottom. The first time you run the ingestion cell, it will create a 'chroma' database folder. You can then ask questions in the final cells.