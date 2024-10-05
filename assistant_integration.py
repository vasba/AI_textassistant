from langchain_huggingface import HuggingFacePipeline
from llama_index import VectorStoreIndex
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

def format_docs(docs):
    return "\n\n".join(doc.node.text for doc in docs)

def generate_response(doc_id: str, question: str, retriever) -> str:
    retrieved_docs = retriever.retrieve(question)
    document_text = format_docs(retrieved_docs)
    #document_text = get_document_text(doc_id)
    if not document_text:
        return "Document not found."
    
    # Tokenize the document text and question
    document_tokens = tokenizer.encode(document_text)
    chunk = ''
    prompt = getPrompt(question, chunk)
    prompt_tokens = tokenizer.encode(prompt)
    # Calculate the maximum input length for the document text
    max_input_length = 2048 - len(prompt_tokens) - 10  # Adjust based on prompt structure
    
    # Split the document text into chunks
    chunks = [document_tokens[i:i + max_input_length] for i in range(0, len(document_tokens), max_input_length)]
    
    responses = []
    for chunk in chunks:
        input_text = getPrompt(question, tokenizer.decode(chunk))	
        response = langchain_pipeline.invoke(input_text, max_new_tokens=300)
        responses.append( response.split("Bot:")[-1].strip())
    
    # Combine the responses
    combined_response = " ".join(responses)
    
    return combined_response

def getPrompt(question, chunk):
    return f"""
        <|endoftext|><s>
        User:
        {chunk}
        {question}
        <s>
        Bot:
        """.strip()