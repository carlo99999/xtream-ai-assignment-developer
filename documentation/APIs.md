# FastAPI Application for Diamond Price Prediction Documentation

This documentation provides an overview and explanation of a FastAPI application designed to handle diamond price prediction. The application allows for data uploading, model training, prediction, and retrieval of similar diamonds.

## Overview

The application uses FastAPI for creating the API endpoints, SQLAlchemy for database interactions, and various machine learning libraries for model training and prediction.

## Imports and Setup

The necessary libraries and modules are imported at the beginning of the script. Each import serves a specific purpose in the application's functionality.

```python
from fastapi import FastAPI, Depends, UploadFile, File, Form, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from DiamondModels import DiamondModel, modelling_algorithms
import io
from Models import SavedDatas, Base, engine, ToPredict, Predicted, SimilarDiamond, Diamond
import uvicorn
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd
import os
```

### Explanation of Imports
- **FastAPI**: For creating the API endpoints.
- **CORS Middleware**: To handle Cross-Origin Resource Sharing.
- **DiamondModels**: Custom module for handling diamond data and models.
- **SQLAlchemy**: For database ORM (Object Relational Mapping).
- **Pandas**: For data manipulation and analysis.
- **Uvicorn**: For running the FastAPI server.

## Database Setup

The database is initialized using SQLAlchemy, and the metadata is created.

```python
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

## FastAPI Application Initialization

The FastAPI application is created and CORS middleware is added to allow all origins.

```python
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Utility Functions

### Add to Database

Helper function to add data to the database.

```python
def add_to_db(db: Session, data):
    db.add(data)
    db.commit()
    db.refresh(data)
    return data
```

### Get Database Session

Dependency to provide a database session.

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## API Endpoints

### Upload Data Endpoint

Endpoint to upload data for training the model.

```python
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
    if file_name.split(".")[-1] not in ["csv", "xlsx", "json"]:
        return JSONResponse(status_code=400, content={"message": "The file format is not supported. Please upload a file in CSV, EXCEL or JSON format."})
    if file_name.split(".")[-1] == "csv":
        datas = pd.read_csv(file.file)
    elif file_name.split(".")[-1] == "xlsx":
        datas = pd.read_excel(file.file)
    elif file_name.split(".")[-1] == "json":
        datas = pd.read_json(file.file)
    datas.to_csv(f"datas_uploaded/{id}.csv", index=False)

    for column in ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]:
        if column not in datas.columns:
            return JSONResponse(status_code=400, content={"message": "The dataset does not contain the required columns: carat, cut, color, clarity, depth, table, price, x, y, z"})

    datas_info = {"model_id": id, "file_name": file_name}
    datas_saved = SavedDatas(**datas_info)
    add_to_db(db, datas_saved)
    return JSONResponse(status_code=200, content={"message": "Datas uploaded successfully!", "data": datas.head(10).to_dict()})
```

### Train Model Endpoint

Endpoint to train the model with uploaded data.

```python
@app.post("/api/train/")
def train_model(
    model_type: str = Form(...),
    id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    This endpoint is used to train the model.
    """
    datas = db.query(SavedDatas).filter(SavedDatas.model_id == id).first()
    if datas is None:
        return JSONResponse(status_code=400, content={"message": "Datas not found. Please upload the datas before training the model."})
    datas = pd.read_csv(f"datas_uploaded/{id}.csv")
    model = DiamondModel(datas=datas, model=model_type, id=id)
    folder_to_save = "trained_models"
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    model.train_model(folder_to_save)
    return JSONResponse(status_code=200, content={"message": "Model trained successfully!"})
```

### Get Columns for Prediction Endpoint

Endpoint to retrieve columns of the uploaded data for making predictions.

```python
@app.get("/api/get_columns_trained/")
def get_columns(id: str = Query(..., description="The id of the model"), db: Session = Depends(get_db)):
    """
    This endpoint is used to get the columns of the datas you uploaded to make predictions.
    """
    for i in os.listdir('datas_uploaded'):
        if i.split('.')[0] == id:
            datas = pd.read_csv(f"datas_uploaded/{i}")
            datas.drop(columns=["price"], inplace=True)
            columns = datas.select_dtypes(include=["object"]).columns

            columns_values = {}
            for column in columns:
                columns_values[column] = datas[column].unique().tolist()
            for i in datas.columns:
                if i not in columns_values:
                    columns_values[i] = []
            return JSONResponse(status_code=200, content={"message": "Columns retrieved successfully!", "data": columns_values})

    return JSONResponse(status_code=400, content={"message": "Datas not found. Please upload the datas before making predictions."})
```

### Predict Endpoint

Endpoint to make predictions using the trained model.

```python
@app.post("/api/predict/")
def predict(id: str = Query(..., description="The id of the model"), diz: dict = Body(...), db: Session = Depends(get_db)):
    """
    This endpoint is used to make predictions with the trained model.
    """
    directory = diz.get("directory", "default_model")
    diz.pop("directory")
    for i in os.listdir(directory):
        if i.split('_')[0] == id and i.endswith(".pkl"):
            model_name = i.split('_')[1].split('.')[0]
            model = DiamondModel(id=id, folder=directory, model=model_name)
            df = pd.DataFrame([], columns=model.datas_dummies.columns)
            df_tmp = pd.DataFrame(diz, index=[0])
            df_tmp = pd.get_dummies(df_tmp, columns=["cut", "color", "clarity"])
            for i in df_tmp.columns:
                if i in df.columns:
                    df[i] = df_tmp[i]
            diz["model_id"] = id
            to_predict = ToPredict(**diz)
            dbToPredict = add_to_db(db, to_predict)
            df.fillna(False, inplace=True)
            df.drop(columns=["price"], inplace=True)
            prediction = model.predict(df).tolist()
            prediction = [round(i, 2) for i in prediction]
            predicted = Predicted(model_id=id, price=prediction[0], toPredictId=dbToPredict.id)
            dbPredicted = add_to_db(db, predicted)
            return JSONResponse(status_code=200, content={"message": "Prediction made successfully!", "prediction": prediction})
    return JSONResponse(status_code=400, content={"message": "Model not found. Please train the model before making predictions."})
```

### Get Columns of Default Data Endpoint

Endpoint to retrieve columns of default data for making predictions.

```python
@app.get("/api/get_columns_default/")
def get_columns(id: str = Query(..., description="The id of the model"), db: Session = Depends(get_db)):
    """
    This endpoint is used to get the columns of the datas you uploaded to make predictions.
    """
    path = "default_model"
    datas_path = f"{path}/Default_datas.csv"
    if os.path.exists(datas_path):
        datas = pd.read_csv(datas_path)
        datas.drop(columns=["price"], inplace=True)
        columns = datas.select_dtypes(include=["object"]).columns
        columns_values = {}
        for column in columns:
            columns_values[column] = datas[column].unique().tolist()
        for i in datas.columns:
            if i not in columns_values:
                columns_values[i] = []
        return JSONResponse(status_code=200, content={"message": "Columns retrieved successfully!", "data": columns_values})

    return JSONResponse(status_code=400, content={"message": "Datas not found. Please upload the datas before making predictions."})
```

### Get Similar Diamonds Endpoint

Endpoint to retrieve similar diamonds based on specified attributes.

```python
@app.post("/api/similar_diamonds")
def get_similar_diamonds(diz: dict = Body(...), db: Session = Depends(get_db)):
    """
    This endpoint is used to get similar diamonds.
    """
    for i in os.listdir('datas_uploaded'):
        n = diz.get("n", 10)
        cut = diz.get("cut

", None)
        color = diz.get("color", None)
        clarity = diz.get("clarity", None)
        carat = diz.get("carat", None)
        diamond = Diamond(carat=carat, cut=cut, color=color, clarity=clarity)
        dbDiamond = add_to_db(db, diamond)
        if carat is not None and cut is not None and color is not None and clarity is not None:
            datas = pd.read_csv(f"datas_uploaded/{i}")
            datas.drop(columns=["price"], inplace=True)
            model = DiamondModel(id="Default", model="XGBRegressor", folder="default_model")
            similar_diamonds = model.get_similar_samples(n=n, cut=cut, color=color, clarity=clarity, carat=carat)
            similar_diamonds = similar_diamonds.to_dict(orient="records")
            for i in similar_diamonds:
                similar_diamond = SimilarDiamond(diamondId=dbDiamond.id, **i)
                add_to_db(db, similar_diamond)
            return JSONResponse(status_code=200, content={"message": "Similar diamonds retrieved successfully!", "data": similar_diamonds})
    return JSONResponse(status_code=400, content={"message": "Datas not found. Please upload the datas before making predictions."})
```

## Running the Application

To run the FastAPI application, use the following command:

```python
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
```

This will start the server on `localhost` at port `8000`.

## Conclusion

This documentation provides an in-depth overview of the FastAPI application for diamond price prediction. The application includes endpoints for uploading data, training models, making predictions, and finding similar diamonds, leveraging FastAPI's capabilities along with machine learning techniques for robust functionality.