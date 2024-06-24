import pandas as pd
from typing import Union
from DiamondModels import DiamondModel
from datetime import datetime
import sys



class Pipeline:
    """
    A pipeline to train a model and make predictions.

    Params:
    datas: Union[pd.DataFrame, str]: The data to train the model, either a pandas DataFrame or a path to a file.
    model: str: The model to train the data. Options:
        - LinearRegression
        - XGBRegressor
    """
    def __init__(self, datas: Union[pd.DataFrame, str], model: str) -> None:
        self.datas = datas
        self.model = model

    def train(self) -> pd.Series:
        """
        Train the model and make predictions.

        Returns:
        pd.Series: Predicted prices.
        """
        self.model_trained = DiamondModel(datas=self.datas, model=self.model)
        self.model_trained.clean_data()
        self.model_trained.train_model()
        
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict prices for new data.

        Params:
        data: DataFrame containing the features for prediction.

        Returns:
        pd.Series: Predicted prices.
        """
        return self.model_trained.predict(data)
    
    def run(self,data: pd.DataFrame) -> pd.Series:
        """
        Train the model and make predictions.

        Returns:
        pd.Series: Predicted prices.
        """
        
        self.train()
        self.prediction=self.predict(data)
        data["price"]=self.prediction
        
        data.to_csv(f"predicted_prices_{datetime.now()}.csv",index=False)
        return data["price"]
    


if __name__ == "__main__":
    model_type=sys.argv[1]
    data_to_train=sys.argv[2]
    pipeline=Pipeline(datas=data_to_train,model=model_type)
    pipeline.train()
    