from datetime import datetime, timezone
from sqlalchemy import DateTime, create_engine, Column, Integer, String, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
Base.metadata.create_all(bind=engine)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    text = Column(String)
    upload_date = Column(DateTime, default=datetime.now(timezone.utc))
    file_size = Column(Integer)
    author = Column(String, nullable=True)
    title = Column(String, nullable=True)

def get_document_text(doc_id: int) -> str:
    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    db.close()
    if doc:
        return doc.text
    return ""