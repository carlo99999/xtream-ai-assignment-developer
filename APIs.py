from fastapi import FastAPI, Depends, UploadFile, File, Form,Query,Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from DiamondModels import DiamondModel,modelling_algorithms
import io
from Models import SavedDatas,Base,engine,ToPredict,Predicted
import uvicorn
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd
import os

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
    return data

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
    
    
@app.post("/api/train/")
def train_model(
    model_type:str=Form(...),
    id:str=Form(...),
    db: Session = Depends(get_db)
):
    """
    This endpoint is used to train the model.
    """
    ### Check if the datas has been uploaded
    datas=db.query(SavedDatas).filter(SavedDatas.model_id==id).first()
    if datas is None:
        return JSONResponse(status_code=400,content={"message":"Datas not found. Please upload the datas before training the model."})
    datas=pd.read_csv(f"datas_uploaded/{id}.csv")
    model=DiamondModel(datas=datas,model=model_type,id=id)
    folder_to_save="trained_models"
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    model.train_model(folder_to_save)
    return JSONResponse(status_code=200,content={"message":"Model trained successfully!"})

@app.get("/api/get_columns_trained/")
def get_columns(id:str=Query(...,description="The id of the model"),db: Session = Depends(get_db)):
    """
    This endpoint is used to get the columns of the datas you uploaded to make predictions.
    """
    for i in os.listdir('datas_uploaded'):
        if i.split('.')[0]==id:
            datas=pd.read_csv(f"datas_uploaded/{i}")
            datas.drop(columns=["price"],inplace=True)
            ## Let's get all the value for a possible column that has dytype object
            columns=datas.select_dtypes(include=["object"]).columns
            
            columns_values={}
            for column in columns:
                columns_values[column]=datas[column].unique().tolist()
            for i in datas.columns:
                if i not in columns_values:
                    columns_values[i]=[]
            return JSONResponse(status_code=200,content={"message":"Columns retrieved successfully!", "data":columns_values})
    
    return JSONResponse(status_code=400,content={"message":"Datas not found. Please upload the datas before making predictions."})

@app.post("/api/predict/")
def predict(id:str=Query(...,description="The id of the model"),diz:dict=Body(...),db: Session = Depends(get_db)):
    """
    This endpoint is used to make predictions with the trained model.
    """
    directory=diz.get("directory","default_model")
    diz.pop("directory")
    for i in os.listdir(directory):
        if i.split('_')[0]==id and i.endswith(".pkl"):
            model_name=i.split('_')[1].split('.')[0]
            print(model_name)
            model=DiamondModel(id=id,folder=directory,model=model_name)
            df=pd.DataFrame([],columns=model.datas_dummies.columns)
            df_tmp=pd.DataFrame(diz,index=[0])
            df_tmp=pd.get_dummies(df_tmp,columns=["cut","color","clarity"])
            for i in df_tmp.columns:
                if i in df.columns:
                    df[i]=df_tmp[i]
            ### Save diz into the db
            diz["model_id"]=id
            to_predict=ToPredict(**diz)
            dbToPredict=add_to_db(db,to_predict)
            df.fillna(False,inplace=True)
            df.drop(columns=["price"],inplace=True)
            prediction=model.predict(df).tolist()
            prediction=[round(i,2) for i in prediction]
            predicted=Predicted(model_id=id,price=prediction[0],toPredictId=dbToPredict.id)
            dbPredicted=add_to_db(db,predicted)
            return JSONResponse(status_code=200,content={"message":"Prediction made successfully!", "prediction":prediction})
    return JSONResponse(status_code=400,content={"message":"Model not found. Please train the model before making predictions."})

@app.get("/api/get_columns_default/")
def get_columns(id:str=Query(...,description="The id of the model"),db: Session = Depends(get_db)):
    """
    This endpoint is used to get the columns of the datas you uploaded to make predictions.
    """
    path="default_model"
    datas_path=f"{path}/Default_datas.csv"
    if os.path.exists(datas_path):
        datas=pd.read_csv(datas_path)
        datas.drop(columns=["price"],inplace=True)
        ## Let's get all the value for a possible column that has dytype object
        columns=datas.select_dtypes(include=["object"]).columns
        columns_values={}
        for column in columns:
            columns_values[column]=datas[column].unique().tolist()
        for i in datas.columns:
            if i not in columns_values:
                columns_values[i]=[]
        return JSONResponse(status_code=200,content={"message":"Columns retrieved successfully!", "data":columns_values})
    
    return JSONResponse(status_code=400,content={"message":"Datas not found. Please upload the datas before making predictions."})

@app.post("/api/similar_diamonds")
def get_similar_diamonds(diz: dict=Body(...),db: Session = Depends(get_db)):
    """
    This endpoint is used to get similar diamonds.
    """
    for i in os.listdir('datas_uploaded'):
        n=diz.get("n",10)
        cut=diz.get("cut",None)
        color=diz.get("color",None)
        clarity=diz.get("clarity",None)
        carat=diz.get("carat",None)
        if carat is not None and cut is not None and color is not None and clarity is not None:
            datas=pd.read_csv(f"datas_uploaded/{i}")
            datas.drop(columns=["price"],inplace=True)
            model=DiamondModel(id="Default",model="XGBRegressor",folder="default_model")
            similar_diamonds=model.get_similar_samples(n=n,cut=cut,color=color,clarity=clarity,carat=carat)
            return JSONResponse(status_code=200,content={"message":"Similar diamonds retrieved successfully!", "data":similar_diamonds.to_dict()})
    return JSONResponse(status_code=400,content={"message":"Datas not found. Please upload the datas before making predictions."})

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)