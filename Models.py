from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float


SQLALCHEMY_DATABASE_URL = "sqlite:///./diamonds.db"

engine=create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

Base=declarative_base()

class SavedDatas(Base):
    __tablename__="SavedDatas"
    id=Column(Integer, primary_key=True, index=True)
    file_name=Column(String)
    model_id=Column(String)
    

