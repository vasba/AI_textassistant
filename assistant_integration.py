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
    document_tokens = tokenizer.encode(document_text)
    prompt = f"""
        <|endoftext|><s>
        User:
        {question}
        <s>
        Bot:
        """.strip()
    prompt_tokens = tokenizer.encode(prompt)
    # Calculate the maximum input length for the document text
    max_input_length = 2048 - len(prompt_tokens) - 10  # Adjust based on prompt structure
    
    # Split the document text into chunks
    chunks = [document_tokens[i:i + max_input_length] for i in range(0, len(document_tokens), max_input_length)]
    
    responses = []
    for chunk in chunks:
        input_text = tokenizer.decode(chunk) + prompt	
        response = langchain_pipeline.invoke(input_text, max_new_tokens=300)
        responses.append( response.split("Bot:")[-1].strip())
    
    # Combine the responses
    combined_response = " ".join(responses)
    
    return combined_response