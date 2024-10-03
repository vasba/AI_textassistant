from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from assistant_integration import generate_response
from db import Base, Document, SessionLocal, engine
from extract_text import extract_text_from_pdf, extract_text_with_ocr

from fastapi import FastAPI, Form, Request, UploadFile, File

Base.metadata.create_all(bind=engine)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

def save_document(file: UploadFile, text: str, author: str, title: str):
    file_size = len(file.file.read())
    file.file.seek(0)  # Reset file pointer after reading size
    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    db = SessionLocal()
    doc = Document(
        filename=file.filename,
        text=text,
        file_size=file_size,
        author=author,
        title=title
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

@app.post("/upload_pdf/")
async def upload_file(file: UploadFile = File(...),
                      author: str = Form(default="Unknown"),
                      title: str = Form(default="Unknown")):
    contents = await file.read()
    text = extract_text_from_pdf(file.filename)
    doc = save_document(file, text, author, title)
    return {"filename": doc.filename, "text": doc.text, "file_size": doc.file_size, "author": doc.author, "title": doc.title}

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
    db = SessionLocal()
    documents = db.query(Document).all()
    db.close()
    return templates.TemplateResponse("index.html", {"request": request, "documents": documents})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, doc_id: int = Form(...), question: str = Form(...)):
    response = generate_response(doc_id, question)
    db = SessionLocal()
    documents = db.query(Document).all()
    db.close()
    return templates.TemplateResponse("index.html", {"request": request, "documents": documents, "response": response, "selected_doc_id": doc_id, "question": question})

