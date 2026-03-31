import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from .config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    tier = Column(String, default="Researcher")
    google_scholar = Column(String, default="")
    institution = Column(String, default="")
    bio = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    login_count = Column(Integer, default=0)


class Study(Base):
    __tablename__ = "studies"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    name = Column(String, nullable=False)
    modality = Column(String, default="EEG")
    subject_count = Column(Integer, default=1)
    conditions = Column(Text, default="")
    sfreq = Column(Float, nullable=True)
    reference = Column(String, default="average")
    notes = Column(Text, default="")
    file_path = Column(String, nullable=True)
    files_json = Column(Text, nullable=True) # JSON list of file info
    file_format = Column(String, nullable=True)
    n_channels = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id = Column(Integer, primary_key=True)
    study_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=True)
    status = Column(String, default="pending")
    pipeline_json = Column(Text, nullable=True)
    results_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    study_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    model = Column(String, default="claude")
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
