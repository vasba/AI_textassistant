from langchain import HuggingFacePipeline
from transformers import pipeline
from sqlalchemy import create_engine

from db import get_document_text

# Load a model from Hugging Face, gpt-2 an alternative
model = pipeline('text-generation', model='sentence-transformers/all-MiniLM-L6-v2')

# Initialize LangChain with the model
langchain_pipeline = HuggingFacePipeline(pipeline=model)

def generate_response(doc_id: int, question: str) -> str:
    document_text = get_document_text(doc_id)
    if not document_text:
        return "Document not found."
    
    prompt = f"Document: {document_text}\n\nQuestion: {question}\nAnswer:"
    return langchain_pipeline(prompt)