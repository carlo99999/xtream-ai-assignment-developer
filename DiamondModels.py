import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib
import uuid
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import optuna
import logging
from typing import Union, List, Dict
import numpy as np
from abc import ABC, abstractmethod
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import logging
from typing import Any, Optional,List
from config import DATA_FOLDER, MODELS_FOLDER, VISUALIZATIONS_FOLDER,COLUMNS_TO_DROP,COLUMNS_TO_DUMMIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)

class BaseModel(ABC):
    """
    Abstract base class for models.
    """
    def __init__(self) -> None:
        self.model_trained: Any = None
        self.mae_mse: Optional[Dict[str, float]] = None
        self.params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Params:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the model.
        
        Params:
        X (pd.DataFrame): Features to predict.

        Returns:
        pd.Series: Predictions.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Params:
        path (str): Path to save the model.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from a file.
        
        Params:
        path (str): Path to load the model.
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Params:
        X (pd.DataFrame): Features to evaluate.
        y (pd.Series): True values.

        Returns:
        Dict[str, float]: Evaluation metrics.
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """
        Get model parameters.
        
        Returns:
        Dict[str, float]: Model parameters.
        """
        pass
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate MAE and MSE metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {"mae": mae, "mse": mse}

class LinearRegressionModel(BaseModel):
    """
    Wrapper for the LinearRegression model from scikit-learn.
    """
    def __init__(self, id: str = None) -> None:
        """
        Initialize the LinearRegressionModel.
        
        Params:
        id (str): Optional model ID to load a saved model.
        """
        if id is None:
            self.model = LinearRegression()
        else:
            self.load(id)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the LinearRegression model.
        
        Params:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        """
        self.model_trained = self.model
        y_log_train = np.log(y_train)
        self.model_trained.fit(X_train, y_log_train)
        
    def predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Predict using the LinearRegression model.
        
        Params:
        X (Union[pd.DataFrame, pd.Series]): Features to predict.

        Returns:
        pd.Series: Predictions.
        """
        
        if self.model_trained is None:
            raise ValueError("Model has not been trained yet.")
        return pd.Series(np.exp(self.model_trained.predict(X)))
    
    def save(self, path: str) -> None:
        """
        Save the LinearRegression model to a file.
        
        Params:
        path (str): Path to save the model.
        """
        joblib.dump(self.model_trained, path)
    
    def load(self, path: str) -> None:
        """
        Load the LinearRegression model from a file.
        
        Params:
        path (str): Path to load the model.
        """
        self.model_trained = joblib.load(path)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the LinearRegression model.
        
        Params:
        X (pd.DataFrame): Features to evaluate.
        y (pd.Series): True values.

        Returns:
        Dict[str, float]: Evaluation metrics.
        """
        y_pred = np.exp(self.predict(X))
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        self.mae_mse = {"mae": mae, "mse": mse}
        return self.mae_mse
    
    def get_params(self) -> Dict[str, float]:
        """
        Get the LinearRegression model parameters.
        
        Returns:
        Dict[str, float]: Model parameters.
        """
        params = {'fit_intercept': self.model_trained.fit_intercept,
                  'copy_X': self.model_trained.copy_X,
                  'n_jobs': self.model_trained.n_jobs, **self.model_trained.get_params()}
        self.params = params
        return params

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int = 5):
        """
        Multi-Layer Perceptron (MLP) model.

        :param input_dim: Dimension of the input features.
        :param hidden_dim: Dimension of the hidden layers.
        :param output_dim: Dimension of the output layer.
        :param num_hidden_layers: Number of hidden layers in the model.
        """
        super(MLP, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
class MLPModel(BaseModel):
    """
    Wrapper for the MLP model using PyTorch.
    """
    def __init__(self, input_dim: int = 19, hidden_dim: int = 128, output_dim: int = 1, id: str = None) -> None:
        """
        Initialize the MLPModel.
        
        Params:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units.
        output_dim (int): Number of output units.
        id (str): Optional model ID to load a saved model.
        """
        super().__init__()
        if id is None:
            self.model = MLP(input_dim, hidden_dim, output_dim)
        else:
            self.load(id)
        self.scaler = None
        self.mae_mse = None
        
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, epochs: int = 1000, lr: float = 0.001) -> None:
        """
        Train the MLP model.
        
        Params:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        """
        self.model_trained = self.model
        self.scaler=StandardScaler()
        optimizer = optim.Adam(self.model_trained.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_train=self.scaler.fit_transform(y_train.reshape(-1,1))
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        self.lr = lr
        self.epoch = epochs
        
        for epoch in range(epochs):
            self.model_trained.train()
            optimizer.zero_grad()
            y_pred = self.model_trained(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
    def predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Predict using the MLP model.
        
        Params:
        X (Union[pd.DataFrame, pd.Series]): Features to predict.

        Returns:
        pd.Series: Predictions.
        """
        self.model_trained.eval()
        X = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.model_trained(X)
        y_pred=self.scaler.inverse_transform(y_pred.numpy())
        return pd.Series(y_pred.flatten())
    
    def save(self, path: str) -> None:
        """
        Save the MLP model to a file.
        
        Params:
        path (str): Path to save the model.
        """
        torch.save(self.model_trained.state_dict(), path)
        joblib.dump(self.scaler, f"{path.split('.')[0]}_scaler.pkl")
        
    def load(self, path: str) -> None:
        """
        Load the MLP model from a file.
        
        Params:
        path (str): Path to load the model.
        """
        self.model = MLP(19, 128, 1)
        if path.endswith('scaler.pkl'):
            self.scaler = joblib.load(path)
            
        else:
            self.model.load_state_dict(torch.load(path))
            self.model_trained = self.model
            
        
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the MLP model.
        
        Params:
        X (pd.DataFrame): Features to evaluate.
        y (pd.Series): True values.

        Returns:
        Dict[str, float]: Evaluation metrics.
        """
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        self.mae_mse = {"mae": mae, "mse": mse}
        return self.mae_mse
    
    def get_params(self) -> Dict[str, float]:
        """
        Get the MLP model parameters.
        
        Returns:
        Dict[str, float]: Model parameters.
        """
        params = {"hidden_units": self.model.input_layer.out_features,
                  "learning_rate": self.lr, **self.mae_mse}
        self.params = params
        return params

class XGBRegressorModel(BaseModel):
    """
    Wrapper for the XGBRegressor model from xgboost.
    """
    def __init__(self, id: str = None) -> None:
        """
        Initialize the XGBRegressorModel.
        
        Params:
        id (str): Optional model ID to load a saved model.
        """
        if id is None:
            self.model = XGBRegressor
        else:
            self.load(id)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the XGBRegressor model.
        
        Params:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        """
        best_params = XGBRegressorModel.find_best_hyperparameters(X_train, y_train, n_trials=100)
        self.best_params = best_params
        self.model_trained = self.model(**best_params, enable_categorical=True, random_state=42)
        self.model_trained.fit(X_train, y_train)
        
    def predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Predict using the XGBRegressor model.
        
        Params:
        X (Union[pd.DataFrame, pd.Series]): Features to predict.

        Returns:
        pd.Series: Predictions.
        """
        return self.model_trained.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the XGBRegressor model to a file.
        
        Params:
        path (str): Path to save the model.
        """
        joblib.dump(self.model_trained, path)
        
        
    def load(self, path: str) -> None:
        """
        Load the XGBRegressor model from a file.
        
        Params:
        path (str): Path to load the model.
        """
        self.model_trained = joblib.load(path)
        
    def evaluate(self, X_test: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the XGBRegressor model.
        
        Params:
        X_test (pd.DataFrame): Features to evaluate.
        y (pd.Series): True values.

        Returns:
        Dict[str, float]: Evaluation metrics.
        """
        y_pred = self.model_trained.predict(X_test)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        self.mae_mse = {"mae": mae, "mse": mse}
        return self.mae_mse
    
    def get_params(self) -> Dict[str, float]:
        """
        Get the XGBRegressor model parameters.
        
        Returns:
        Dict[str, float]: Model parameters.
        """
        if hasattr(self, 'best_params'):
            params = {**self.best_params, 'n_estimators': self.model_trained.n_estimators, **self.mae_mse}
        else:
            params = {'n_estimators': self.model_trained.n_estimators, **self.mae_mse}
        self.params = params
        return params

    @staticmethod
    def objective(x_train_xgb: pd.DataFrame, y_train_xgb: pd.Series, trial: optuna.trial.Trial) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Params:
        x_train_xgb (pd.DataFrame): Training features for XGBoost.
        y_train_xgb (pd.Series): Training labels for XGBoost.
        trial (optuna.trial.Trial): Optuna trial object.

        Returns:
        float: Mean absolute error on validation set.
        """
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }

        x_train, x_val, y_train, y_val = train_test_split(x_train_xgb, y_train_xgb, test_size=0.2, random_state=42)
        model = XGBRegressor(**param)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    @staticmethod
    def find_best_hyperparameters(x_train_xgb: pd.DataFrame, y_train_xgb: pd.Series, n_trials: int = 100) -> dict:
        """
        Find the best hyperparameters for XGBoost using Optuna.

        Params:
        x_train_xgb (pd.DataFrame): Training features for XGBoost.
        y_train_xgb (pd.Series): Training labels for XGBoost.
        n_trials (int): Number of trials for Optuna optimization.

        Returns:
        dict: Best hyperparameters found.
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: XGBRegressorModel.objective(x_train_xgb, y_train_xgb, trial), n_trials=n_trials)
        return study.best_params

    


file_readers = {
    '.csv': pd.read_csv,
    '.xlsx': pd.read_excel,
    '.xls': pd.read_excel,
    '.json': pd.read_json,
    '.html': pd.read_html,
    '.hdf': pd.read_hdf,
    '.feather': pd.read_feather,
    '.parquet': pd.read_parquet,
    '.pkl': pd.read_pickle,
    '.sql': pd.read_sql,
    '.stata': pd.read_stata,
    '.sas': pd.read_sas,
    '.spss': pd.read_spss,
    '.dta': pd.read_stata,
    '.orc': pd.read_orc,
    '.gbq': pd.read_gbq
}

modelling_algorithms = {
    "XGBRegressor": XGBRegressorModel,
    "LinearRegression": LinearRegressionModel,
    "MLP": MLPModel   ## Added this to prove the point that my classes are pretty flexible
}

class DiamondModel:
    """
    Class for managing diamond price prediction models.
    """
    def __init__(self, datas: Union[pd.DataFrame, str] = None, id: str = None, model: str = None, folder: str = MODELS_FOLDER) -> None:
        """
        Initialize the DiamondModel.

        Params:
        datas (Union[pd.DataFrame, str]): The data to train the model, either a pandas DataFrame or a path to a file.
        id (str): The ID of the model to load.
        model (str): The model to train the data. Options: 'LinearRegression', 'XGBRegressor'.
        folder (str): The folder where models and data are saved/loaded.
        """
        if datas is None and id is None:
            raise ValueError("You must provide data to train the model or an id to load the model")
        if id is None and model is None:
            raise ValueError("You must decide a model to train the data")
        if model and model not in modelling_algorithms:
            raise ValueError("The model provided is not yet supported")
        
        self.folder = folder
        
        if id is None and datas is not None:
            self._create_new_model(datas, folder, model)
        elif id and datas is None:
            self._load(id)
        elif id and datas is not None:
            if not self._check_if_model_exists(id):
                self._create_new_model(datas, folder, model, id)

    def _check_if_model_exists(self, id: str) -> bool:
        """
        Check if a model with the given ID already exists.

        Params:
        id (str): Model ID.

        Returns:
        bool: True if the model exists, False otherwise.
        """
        for i in os.listdir(self.folder):
            if i.startswith(id):
                raise ValueError("A model with the same id already exists")
        return False

    def _create_new_model(self, datas: Union[pd.DataFrame, str], folder: str, model: str, id: str = None) -> None:
        """
        Create a new model.

        Params:
        datas (Union[pd.DataFrame, str]): The data to train the model.
        folder (str): The folder where models and data are saved/loaded.
        model (str): The model to train the data.
        id (str): Optional model ID.
        """
        self.id = uuid.uuid4().hex if id is None else id
        self.model_name = model
        self.model: BaseModel = modelling_algorithms.get(model)()

        if isinstance(datas, str):
            self.datas = self._load_data_from_file(datas)
        elif isinstance(datas, pd.DataFrame):
            self.datas = datas

        if self.datas is None and datas is not None:
            raise ValueError("The data could not be loaded")

    @staticmethod
    def plot_gof(y_true: pd.Series, y_pred: pd.Series,show: bool,path:str=VISUALIZATIONS_FOLDER,save: bool=False) -> None:
        """
        Plot the goodness of fit.

        Params:
        y_true (pd.Series): Actual values.
        y_pred (pd.Series): Predicted values.
        """
        fig=px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_shape(type='line', x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(), line=dict(color='red', width=2))
        fig.update_layout(title='Goodness of Fit', xaxis_title='Actual', yaxis_title='Predicted')
        if save:
            if not os.path.exists(path):
                os.makedirs(path)

            fig.write_image(f'{path}/goodness_of_fit.png')
        if show:
            plt.show()
        return fig

    @staticmethod
    def prepare_data_for_prediction(data_blueprint: pd.DataFrame, data_to_predict: pd.DataFrame, target_column: str = "price") -> pd.DataFrame:
        """
        Prepares the data to be predicted by ensuring it has the same columns as the training data.

        Params:
        data_blueprint (pd.DataFrame): The dataframe blueprint with the expected columns.
        data_to_predict (pd.DataFrame): The new data to prepare for prediction.
        target_column (str): The target column to exclude from prediction data.

        Returns:
        pd.DataFrame: The prepared dataframe with the same columns as the blueprint.
        """
        ## Check if the target column is in the data blueprint
        if target_column in data_blueprint.columns:
            data_blueprint = data_blueprint.drop(columns=[target_column])
        ## Check if the columns are the same,if they are the same, return the data_to_predict
        if list(data_to_predict.columns) == list(data_blueprint.columns):
            return data_to_predict
        
        blueprint_columns = set(col.split('_')[0] for col in data_blueprint.columns)
        data_columns = set(data_to_predict.columns)
        missing_columns = blueprint_columns - data_columns
        if missing_columns:
            raise ValueError(f"Missing columns in the data you want to predict: {missing_columns}")

        columns_to_dummies = data_to_predict.select_dtypes(include=['object']).columns
        predict = pd.get_dummies(data_to_predict, columns=columns_to_dummies,dtype=float)
        for col in data_blueprint.columns:
            if col not in predict.columns:
                predict[col] = 0

        predict = predict[data_blueprint.columns]
        
            
        return predict

    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.

        Params:
        file_path (str): Path to the file.

        Returns:
        pd.DataFrame: Loaded data.
        """
        try:
            for extension, reader in file_readers.items():
                if file_path.endswith(extension):
                    df = reader(file_path)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    return df
            raise ValueError(f"Unsupported file extension: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def _load(self, id: str) -> None:
        """
        Load a model and its data from files.

        Params:
        id (str): Model ID.
        """
        self.model = None
        self.datas = None
        self.datas_dummies = None
        self.datas_processed = None
        for i in os.listdir(self.folder):
            if i.startswith(id):
                self.id = id
                if i.endswith('.pkl'):
                    model_name = i.split('_')[1].split('.')[0]
                    self.model_name = model_name
                    self.model = modelling_algorithms.get(model_name)() if self.model is None else self.model
                    self.model.load(f'{self.folder}/{i}')
                elif i.endswith('_datas.csv'):
                    self.datas = pd.read_csv(f'{self.folder}/{i}')
                elif i.endswith('_dummies.csv'):
                    self.datas_dummies = pd.read_csv(f'{self.folder}/{i}')
                elif i.endswith('_processed.csv'):
                    self.datas_processed = pd.read_csv(f'{self.folder}/{i}')
        self._validate_loaded_data()

    def _validate_loaded_data(self) -> None:
        """
        Validate that the required data and model components are loaded.
        Raises a ValueError if any component is missing.
        """
        if self.datas is None:
            raise ValueError("The data could not be loaded")
        if self.model is None:
            raise ValueError("The model could not be loaded")
        if self.datas_dummies is None:
            raise ValueError("The dummies could not be loaded")
        if self.datas_processed is None:
            raise ValueError("The processed data could not be loaded")

    def _save(self, folder_to_save: str = None) -> None:
        """
        Save the model and its data to files.

        Params:
        folder_to_save (str): Folder where to save the files.
        """
        folder_to_save = folder_to_save if folder_to_save else self.folder
        os.makedirs(folder_to_save, exist_ok=True)
        self.model.save(f'{folder_to_save}/{self.id}_{self.model_name}.pkl')
        self.datas.to_csv(f'{folder_to_save}/{self.id}_datas.csv', index=False)
        self.datas_dummies.to_csv(f'{folder_to_save}/{self.id}_dummies.csv', index=False)
        self.datas_processed.to_csv(f'{folder_to_save}/{self.id}_processed.csv', index=False)
        with open(f'{folder_to_save}/{self.id}_params.json', 'w') as f:
            json.dump(self.params, f)

    def clean_data(self, columns_to_drop: List[str] = COLUMNS_TO_DROP, columns_to_dummies: List[str] = COLUMNS_TO_DUMMIES) -> None:
        """
        Clean the data by dropping specified columns and creating dummy variables for categorical features.

        Params:
        columns_to_drop (List[str]): List of columns to drop.
        columns_to_dummies (List[str]): List of columns to convert to dummy variables.
        """
        self.datas = self.datas.dropna()
        self.datas = self.datas[(self.datas.x * self.datas.y * self.datas.z != 0) & (self.datas.price > 0)].reset_index(drop=True)
        self.datas_processed = self.datas.drop(columns=columns_to_drop)
        
        columns = [col for col in columns_to_dummies if col not in columns_to_drop or col.find('Unnamed') == -1]
        self.datas_dummies = pd.get_dummies(self.datas_processed, columns=columns, drop_first=True,dtype=float)

    def visualize_scatter_matrix(self, save: bool = False,show: bool=True,path:str='visualizations') -> plt.Figure:
        """
        Visualize a scatter matrix of the numerical features.

        Params:
        save (bool): Whether to save the plot to a file.

        Returns:
        plt.Figure: Scatter matrix figure.
        """
        fig = pd.plotting.scatter_matrix(self.datas.select_dtypes(include=['number']), figsize=(14, 10))
        if save:
            os.makedirs(f'{path}', exist_ok=True)
            plt.savefig(f'{path}/{self.id}_scatter_matrix.png')
        if show:
            plt.show()
       
        return fig

    def visualize_histogram(self, save: bool = False,show: bool=True,path:str='visualizations') -> plt.Figure:
        """
        Visualize histograms of the features.

        Params:
        save (bool): Whether to save the plot to a file.

        Returns:
        plt.Figure: Histogram figure.
        """
        fig = self.datas.hist(bins=100, figsize=(14, 10))
        if save:
            os.makedirs(f'{path}', exist_ok=True)
            plt.savefig(f'{path}/{self.id}_histogram.png')
        if show:
            plt.show()
        return fig

    def visualize_diamond_prices_by(self, cut_column: str, save: bool = False,show: bool=True,path:str='visualizations') -> px.violin:
        """
        Visualize diamond prices by a categorical feature using a violin plot.

        Params:
        cut_column (str): The categorical column to plot.
        save (bool): Whether to save the plot to a file.

        Returns:
        px.violin: Violin plot figure.
        """
        fig = px.violin(self.datas, x=cut_column, y='price', color=cut_column, title=f'Price by {cut_column}')
        if show:
            fig.show()
        if save:
            os.makedirs(f'{path}', exist_ok=True)
            fig.write_html(f'{path}/{self.id}_price_by_{cut_column}.html')
        return fig

    def train_model(self, folder_to_save: str = None) -> Dict[str, float]:
        """
        Train the model and save it to a file.

        Params:
        folder_to_save (str): Folder where to save the trained model and data.

        Returns:
        Dict[str, float]: Mean absolute error and mean squared error on the test set.
        """
        if not hasattr(self, 'datas_dummies'):
            warnings.warn("The data is not yet cleaned, cleaning it now using the default columns to drop")
            self.clean_data()

        X = self.datas_dummies.drop(columns=['price'])
        y = self.datas_dummies['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.train(X_train=X_train, y_train=y_train)
        self.predictions = self.model.predict(X_test)
        mae_mse = self.model.evaluate(X_test, y_test)
        self.params = self.model.get_params()
        self._save(folder_to_save)
        self.GT_Y = y_test
        return mae_mse

    def plot_predictions_vs_actual(self, save: bool = False,show: bool=False,path=VISUALIZATIONS_FOLDER) -> None:
        """
        Plot the predicted values against the actual values.

        Params:
        save (bool): Whether to save the plot to a file.
        """
        DiamondModel.plot_gof(self.GT_Y, self.predictions,show=show,path=path,save=save)
        

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict prices for new data.

        Params:
        data (pd.DataFrame): DataFrame containing the features for prediction.

        Returns:
        pd.Series: Predicted prices.
        """
        data = DiamondModel.prepare_data_for_prediction(self.datas_dummies, data)
        predictions = self.model.predict(data)
        return predictions

    def get_similar_samples(self, carat: float, cut: str, color: str, clarity: str, n: int) -> pd.DataFrame:
        """
        Get similar diamond samples based on carat, cut, color, and clarity.

        Params:
        carat (float): Carat weight of the diamond.
        cut (str): Cut of the diamond.
        color (str): Color of the diamond.
        clarity (str): Clarity of the diamond.
        n (int): Number of similar samples to return.

        Returns:
        pd.DataFrame: DataFrame containing the similar samples.
        """
        if self.datas is None:
            raise ValueError("Data is not loaded.")
        if carat < 0 or n < 0:
            raise ValueError("Carat and n must be greater than 0")
        if n > len(self.datas):
            return self.datas
        if clarity not in self.datas['clarity'].unique() or color not in self.datas['color'].unique() or cut not in self.datas['cut'].unique():
            raise ValueError("Cut, color, and clarity must be one of the unique values in the dataset")
        filtered_data = self.datas[(self.datas['cut'] == cut) & (self.datas['color'] == color) & (self.datas['clarity'] == clarity)]
        if filtered_data.empty:
            return pd.DataFrame()
        filtered_data['weight_diff'] = (filtered_data['carat'] - carat).abs()
        similar_samples = filtered_data.sort_values(by='weight_diff').head(n)
        return similar_samples.drop(columns=['weight_diff'])
