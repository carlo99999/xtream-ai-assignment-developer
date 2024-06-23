from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float,ForeignKey


SQLALCHEMY_DATABASE_URL = "sqlite:///./diamonds.db"

engine=create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

Base=declarative_base()

class SavedDatas(Base):
    __tablename__="SavedDatas"
    id=Column(Integer, primary_key=True, index=True)
    file_name=Column(String)
    model_id=Column(String)
    
class ToPredict(Base):
    __tablename__="ToPredict"
    id=Column(Integer, primary_key=True, index=True)
    model_id=Column(String)
    carat=Column(Float)
    cut=Column(String)
    color=Column(String)
    clarity=Column(String)
    depth=Column(Float)
    table=Column(Float)
    x=Column(Float)
    y=Column(Float)
    z=Column(Float)
    
    predictions = relationship("Predicted", back_populates="to_predict")
    
    
class Predicted(Base):
    __tablename__="Predicted"
    id=Column(Integer, primary_key=True, index=True)
    toPredictId=Column(Integer, ForeignKey("ToPredict.id"))
    model_id=Column(String)
    price=Column(Float)
    
    to_predict = relationship("ToPredict", back_populates="predictions")
    
    
class Diamond(Base):
    __tablename__ = "Diamonds"
    id = Column(Integer, primary_key=True, index=True)
    carat = Column(Float)
    cut = Column(String)
    color = Column(String)
    clarity = Column(String)
    
    similar_diamonds = relationship("SimilarDiamond", back_populates="diamond")

class SimilarDiamond(Base):
    __tablename__ = "SimilarDiamonds"
    id = Column(Integer, primary_key=True, index=True)
    diamondId = Column(Integer, ForeignKey("Diamonds.id"))
    carat = Column(Float)
    cut = Column(String)
    color = Column(String)
    clarity = Column(String)
    price = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    table = Column(Float)
    depth = Column(Float)
    
    diamond = relationship("Diamond", back_populates="similar_diamonds")