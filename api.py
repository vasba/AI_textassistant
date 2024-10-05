import json
from tempfile import NamedTemporaryFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from assistant_integration import generate_response
from db import Base, Document, SessionLocal, engine
from extract_text import extract_text_from_pdf, extract_text_with_ocr

from transformers import AutoTokenizer
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.storage.storage_context import StorageContext
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters


from fastapi import FastAPI, Form, Request, UploadFile, File

def initialize_components(chunk_size=514, chunk_overlap=100):
    """
    Initializes the components needed for the application.

    Parameters:
    chunk_size (int): The size of the chunks that the text should be split into. Default is 120.
    chunk_overlap (int): The number of characters that should overlap between chunks. Default is 20.

    Returns:
    tuple: A tuple containing the initialized components: 
    - service_context (ServiceContext): The service context.
    - text_splitter (RecursiveCharacterTextSplitter): The text splitter.
    - chroma_collection (Collection): The database collection.
    - storage_context (StorageContext): The storage context.
    - vector_store (ChromaVectorStore): The vector store.
    """
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    embed_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    db2 = chromadb.PersistentClient(path="database/chroma_db")
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
     
    return service_context, text_splitter, chroma_collection, storage_context, vector_store

def get_unique_document_names(input_chroma_collection):
    all_ids = input_chroma_collection.get()

    unique_names = set()  # Use a set to ensure uniqueness

    for item in all_ids['metadatas']: 
        # TODO find a simpler way to get the file name
        node_content = item['_node_content']
        content = json.loads(node_content)
        file_name = content.get('metadata').get('file_name')
        unique_names.add(file_name)

    return list(unique_names)

def index_document(document_path, input_storage_context, input_text_splitter, input_service_context):
    """
    This function reads a document from the specified path, splits it into chunks, 
    generates embeddings for each chunk using the provided embedding model, and stores 
    the embeddings in the provided storage context for later retrieval.

    Parameters:
    document_path (str): The path to the document to index.
    input_storage_context (StorageContext): The storage context to store the embeddings in.
    input_text_splitter (RecursiveCharacterTextSplitter): The text splitter to use to split the document into chunks.
    input_service_context (ServiceContext): The service context that provides the language model and embedding model.
    """  
    uploaded_document = SimpleDirectoryReader(input_files=[document_path]).load_data()
    VectorStoreIndex.from_documents(uploaded_document, service_context=input_service_context,storage_context=input_storage_context, show_progress=True, text_splitter = input_text_splitter )
    return

service_context, text_splitter, chroma_collection, storage_context, vector_store = initialize_components()

document_list = get_unique_document_names(chroma_collection)

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def save_document(file: UploadFile, text: str, author: str, title: str):
    file.file.seek(0)  # Reset file pointer after reading size
    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    index_document(file.filename, storage_context, text_splitter, service_context)
    return file.filename

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload_pdf/")
async def upload_file(file: UploadFile = File(...),
                      author: str = Form(default="Unknown"),
                      title: str = Form(default="Unknown")):
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    filename = save_document(file, text, author, title)
    return {"filename": filename}

@app.post("/upload_ocr/")
async def upload_file_ocr(file: UploadFile = File(...),
                          author: str = Form(default="Unknown"),
                          title: str = Form(default="Unknown")):
    contents = await file.read()
    text = extract_text_with_ocr(file.filename)
    doc = save_document(file, text, author, title)
    return {"filename": doc.filename, "text": doc.text, "file_size": doc.file_size, "author": doc.author, "title": doc.title}


@app.get("/documents/{doc_id}")
def read_document(doc_id: int):
    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    return {"filename": doc.filename, "text": doc.text}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    #db = SessionLocal()
    #documents = db.query(Document).all()
    document_list = get_unique_document_names(chroma_collection)
    #db.close()
    return templates.TemplateResponse("index.html", {"request": request, "documents": document_list})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, doc_id: str = Form(...), question: str = Form(...)):
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    num_results = 5
    num_queries = 1
    filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=doc_id)])
    retriever = index.as_retriever(verbose=True, similarity_top_k=num_results, num_queries = num_queries, filters=filters)
    response = generate_response(doc_id, question, retriever)
    return templates.TemplateResponse("index.html", {"request": request, "documents": document_list, "response": response, "selected_doc_id": doc_id, "question": question})

