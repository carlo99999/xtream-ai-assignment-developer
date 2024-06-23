from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from DiamondModels import DiamondModel,modelling_algorithms
import io
from Models import SavedDatas,Base,engine
import uvicorn
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd

Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def add_to_db(db:Session,data):
    db.add(data)
    db.commit()
    db.refresh(data)

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/datas/")
def upload_datas(
    file_name: str = Form(...),
    id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    This endpoint is used to upload the datas to train the model.
    """
    ### Upload the datas to the server
    if file_name.split(".")[-1] not in ["csv","xlsx","json"]:
        return JSONResponse(status_code=400,content={"message":"The file format is not supported. Please upload a file in CSV, EXCEL or JSON format."})
    if file_name.split(".")[-1]=="csv":
        datas=pd.read_csv(file.file)
    elif file_name.split(".")[-1]=="xlsx":
        datas=pd.read_excel(file.filr)
    elif file_name.split(".")[-1]=="json":
        datas=pd.read_json(file.file)
    datas.to_csv(f"datas_uploaded/{id}.csv",index=False)
    ### Check if the datas contains the required columns
    for column in ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]:
        if column not in datas.columns:
            return JSONResponse(status_code=400,content={"message":"The dataset does not contain the required columns: carat, cut, color, clarity, depth, table, price, x, y, z"})
    
    ## Save the file name and the id to the database to keep track of the datas
    datas_info={"model_id":id,"file_name":file_name}
    datas_saved=SavedDatas(**datas_info)
    add_to_db(db,datas_saved)
    return JSONResponse(status_code=200,content={"message":"Datas uploaded successfully!", "data":datas.head(10).to_dict()})
    
    
    
    
if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)