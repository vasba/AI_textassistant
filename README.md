# AI_textassistant

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd AI_textassistant
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r Requirements.txt
    ```

## Download Model

1. **Download the model:**
    ```sh
    huggingface-cli download AI-Sweden-Models/gpt-sw3-1.3b-instruct
    ```

2. **Ensure the model files are in the correct directory:**
    Place the downloaded model files in a directory accessible by your application.

## Running the Application

1. **Start the FastAPI server:**
    ```sh
    uvicorn api:app --reload
    ```

2. **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:8000`.

## Using the UI

1. **Upload a PDF:**
    - Navigate to the upload page at `http://127.0.0.1:8000/upload`.
    - Select a PDF file, enter the author and title, and click "Upload".

2. **Ask a Question:**
    - After uploading a document, go to the main page.
    - Select a document from the list and enter your question in the provided form.
    - Click "Submit" to get a response generated by the AI model.

## File Structure

- `api.py`: Contains the FastAPI endpoints and application setup.
- `assistant_integration.py`: Integrates the Hugging Face model with LangChain.
- `db.py`: Database setup and utility functions.
- `extract_text.py`: Functions to extract text from PDFs.
- `static/`: Contains static files like CSS.
- `templates/`: HTML templates for the web UI.

## Requirements

- Python 3.8+
- Dependencies listed in `Requirements.txt`

# Resources

https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
https://python.langchain.com/docs/tutorials/rag/
https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-qa/question_answering_documents_langchain.ipynb#scrollTo=ZmkGiHXB2ID3
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb

Also you can search on Github for the model of intrest for see other's trial