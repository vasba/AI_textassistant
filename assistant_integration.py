from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from db import get_document_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
prompt = "Träd är fina för att"

model_name = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
model.eval()
model.to(device)

# Load a model from Hugging Face, gpt-2 as an alternative
model = pipeline('text-generation', tokenizer=tokenizer, model=model, device=device)

# Initialize LangChain with the model
langchain_pipeline = HuggingFacePipeline(pipeline=model)

def generate_response(doc_id: int, question: str) -> str:
    document_text = get_document_text(doc_id)
    if not document_text:
        return "Document not found."
    
    # Tokenize the document text and question
    question_tokens = tokenizer.encode(question)
    document_tokens = tokenizer.encode(document_text)
    
    # Calculate the maximum input length for the document text
    max_input_length = 2048 - len(question_tokens) - 10  # Adjust based on prompt structure
    
    # Split the document text into chunks
    chunks = [document_tokens[i:i + max_input_length] for i in range(0, len(document_tokens), max_input_length)]
    
    responses = []
    for chunk in chunks:
        input_tokens = chunk + question_tokens
        input_text = tokenizer.decode(input_tokens)
        response = langchain_pipeline.invoke(input_text, max_new_tokens=50)
        responses.append(response[0]['generated_text'])
    
    # Combine the responses
    combined_response = " ".join(responses)
    
    return combined_response