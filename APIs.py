from fastapi import FastAPI, Depends, UploadFile, File, Form, Query, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import pandas as pd
import os
import logging
from sqlalchemy.orm import Session, sessionmaker
from typing import List, Dict, Generator, Any
from Models import SavedDatas, Base, engine, ToPredict, Predicted, SimilarDiamond, Diamond
from DiamondModels import DiamondModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-allowed-origin.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def add_to_db(db: Session, data: Any) -> Any:
    """
    Adds the given data to the database session, commits the transaction, refreshes the instance, and returns it.

    :param db: Database session.
    :param data: Data to be added to the database.
    :return: The added and refreshed instance.
    """
    db.add(data)
    db.commit()
    db.refresh(data)
    return data

def get_db() -> Generator:
    """
    Dependency that provides a SQLAlchemy database session.

    :return: SQLAlchemy database session generator.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def read_file(file: UploadFile) -> pd.DataFrame:
    """
    Reads the uploaded file and returns a pandas DataFrame.

    :param file: The uploaded file.
    :return: A pandas DataFrame containing the file data.
    """
    try:
        if file.filename.endswith(".csv"):
            return pd.read_csv(file.file)
        elif file.filename.endswith(".xlsx"):
            return pd.read_excel(file.file)
        elif file.filename.endswith(".json"):
            return pd.read_json(file.file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV, Excel or JSON file.")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

@app.post("/api/datas/")
async def upload_datas(
    file_name: str = Form(...),
    id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Uploads the data to train the model.

    :param file_name: The name of the file being uploaded.
    :param id: The ID for the uploaded data.
    :param file: The uploaded file.
    :param db: Database session dependency.
    :return: JSONResponse indicating the status of the upload.
    """
    try:
        if file_name.split(".")[-1] not in ["csv", "xlsx", "json"]:
            return JSONResponse(status_code=400, content={"message": "The file format is not supported. Please upload a file in CSV, EXCEL or JSON format."})
        
        datas = read_file(file)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    
    for col in datas.columns:
        if "Unnamed" in col:
            datas.drop(columns=[col], inplace=True)
    
    os.makedirs("datas_uploaded", exist_ok=True)
    datas.to_csv(f"datas_uploaded/{id}.csv", index=False)

    required_columns = {"carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"}
    if not required_columns.issubset(datas.columns):
        missing_cols = required_columns - set(datas.columns)
        return JSONResponse(status_code=400, content={"message": f"The dataset must contain the following columns: {', '.join(missing_cols)}"})
    
    datas_info = {"model_id": id, "file_name": file_name}
    datas_saved = SavedDatas(**datas_info)
    add_to_db(db, datas_saved)

    logger.info(f"Data uploaded successfully: {file_name} with ID: {id}")
    return JSONResponse(status_code=200, content={"message": "Data uploaded successfully!", "data": datas.head(10).to_dict()})

@app.post("/api/train/")
async def train_model(
    model_type: str = Form(...),
    id: str = Form(...),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Trains the model using the uploaded data.

    :param model_type: The type of model to train.
    :param id: The ID of the uploaded data.
    :param db: Database session dependency.
    :return: JSONResponse indicating the status of the training.
    """
    datas_record = db.query(SavedDatas).filter(SavedDatas.model_id == id).first()
    if datas_record is None:
        raise HTTPException(status_code=400, detail="Data not found. Please upload the data before training the model.")

    datas = pd.read_csv(f"datas_uploaded/{id}.csv")
    try:
        model = DiamondModel(datas=datas, model=model_type, id=id)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while initializing the model.")

    folder_to_save = "trained_models"
    os.makedirs(folder_to_save, exist_ok=True)
    model.train_model(folder_to_save)

    logger.info(f"Model trained successfully: {model_type} with ID: {id}")
    return JSONResponse(status_code=200, content={"message": "Model trained successfully!"})

@app.get("/api/get_columns_trained/")
async def get_columns(id: str = Query(..., description="The ID of the model"), db: Session = Depends(get_db)) -> JSONResponse:
    """
    Retrieves the columns of the uploaded data for making predictions.

    :param id: The ID of the model.
    :param db: Database session dependency.
    :return: JSONResponse containing the columns of the uploaded data.
    """
    file_path = f"datas_uploaded/{id}.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Data not found. Please upload the data before making predictions.")

    datas = pd.read_csv(file_path)
    if "price" in datas.columns:
        datas.drop(columns=["price"], inplace=True)

    columns = datas.select_dtypes(include=["object"]).columns
    columns_values = {column: datas[column].unique().tolist() for column in columns}
    for col in datas.columns:
        if col not in columns_values:
            columns_values[col] = []

    logger.info(f"Columns retrieved successfully for model ID: {id}")
    return JSONResponse(status_code=200, content={"message": "Columns retrieved successfully!", "data": columns_values})

@app.post("/api/predict/")
async def predict(
    id: str = Query(..., description="The ID of the model"),
    data: Dict = Body(...),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Makes predictions using the trained model.

    :param id: The ID of the model.
    :param data: The input data for prediction.
    :param db: Database session dependency.
    :return: JSONResponse containing the prediction result.
    """
    directory = data.get("directory", "default_model")
    data.pop("directory", None)

    model_file = next((f for f in os.listdir(directory) if f.startswith(id) and f.endswith(".pkl")), None)
    if not model_file:
        raise HTTPException(status_code=400, detail="Model not found. Please train the model before making predictions.")

    model_name = model_file.split('_')[1].split('.')[0]
    try:
        model = DiamondModel(id=id, folder=directory, model=model_name)
        df = pd.DataFrame([data])

        prediction = model.predict(df)
        prediction = [round(p, 2) for p in prediction]

        data["model_id"] = id
        to_predict = ToPredict(**data)
        db_to_predict = add_to_db(db, to_predict)
        predicted = Predicted(model_id=id, price=prediction[0], toPredictId=db_to_predict.id)
        add_to_db(db, predicted)

        logger.info(f"Prediction made successfully for model ID: {id}")
        return JSONResponse(status_code=200, content={"message": "Prediction made successfully!", "prediction": prediction})
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while making the prediction. {e}")

@app.get("/api/get_columns_default/")
async def get_columns_default(
    id: str = Query(..., description="The ID of the model"),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Retrieves the default columns of the data for making predictions.

    :param id: The ID of the model.
    :param db: Database session dependency.
    :return: JSONResponse containing the default columns of the data.
    """
    path = "default_model"
    datas_path = f"{path}/Default_datas.csv"
    if not os.path.exists(datas_path):
        raise HTTPException(status_code=400, detail="Data not found. Please upload the data before making predictions.")

    datas = pd.read_csv(datas_path)
    if "price" in datas.columns:
        datas.drop(columns=["price"], inplace=True)

    columns = datas.select_dtypes(include=["object"]).columns
    columns_values = {column: datas[column].unique().tolist() for column in columns}
    for col in datas.columns:
        if col not in columns_values:
            columns_values[col] = []

    logger.info(f"Default columns retrieved successfully for model ID: {id}")
    return JSONResponse(status_code=200, content={"message": "Columns retrieved successfully!", "data": columns_values})

@app.post("/api/similar_diamonds/")
async def get_similar_diamonds(
    data: Dict = Body(...),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Retrieves similar diamonds based on the provided attributes.

    :param data: The attributes of the diamond.
    :param db: Database session dependency.
    :return: JSONResponse containing similar diamonds.
    """
    try:
        n = data.get("n", 10)
        cut = data.get("cut")
        color = data.get("color")
        clarity = data.get("clarity")
        carat = data.get("carat")

        if not all([carat, cut, color, clarity]):
            raise HTTPException(status_code=400, detail="All diamond attributes (carat, cut, color, clarity) must be provided.")

        diamond = Diamond(carat=carat, cut=cut, color=color, clarity=clarity)
        db_diamond = add_to_db(db, diamond)
        
        model = DiamondModel(id="Default", model="XGBRegressor", folder="default_model")
        similar_diamonds = model.get_similar_samples(n=n, cut=cut, color=color, clarity=clarity, carat=carat)
        similar_diamonds = similar_diamonds.to_dict(orient="records")

        for diamond_data in similar_diamonds:
            similar_diamond = SimilarDiamond(diamondId=db_diamond.id, **diamond_data)
            add_to_db(db, similar_diamond)

        logger.info(f"Similar diamonds retrieved successfully for attributes: {data}")
        return JSONResponse(status_code=200, content={"message": "Similar diamonds retrieved successfully!", "data": similar_diamonds})
    except Exception as e:
        logger.error(f"Error retrieving similar diamonds: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving similar diamonds. {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
