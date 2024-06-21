import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import joblib
import uuid
import xgboost as xgb
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import optuna
from typing import Union
import json
import numpy as np

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
    "LinearRegression": LinearRegression,
    "XGBRegressor": XGBRegressor
}

class DiamondModel:
    """
    Initialize the model either with data to train the model or an id to load an existing model.
    
    Params:
    datas: Union[pd.DataFrame, str]: The data to train the model, either a pandas DataFrame or a path to a file.
    id: str: The id of the model to load.
    model: str: The model to train the data. Options:
        - LinearRegression
        - XGBRegressor
    """
    def __init__(self, datas: Union[pd.DataFrame, str] = None, id: str = None, model: str = None) -> None:
        if datas is None and id is None:
            raise ValueError("You must provide data to train the model or an id to load the model")
        if id is None and model is None:
            raise ValueError("You must decide a model to train the data")
        if model and model not in modelling_algorithms:
            raise ValueError("The model provided is not yet supported")
        
        self.id = id or uuid.uuid4().hex
        self.model_name = model
        self.model = modelling_algorithms.get(model)
        
        if isinstance(datas, str):
            self.datas = self._load_data_from_file(datas)
        elif isinstance(datas, pd.DataFrame):
            self.datas = datas
        
        if id:
            self._load(id)
        
        if self.datas is None and datas is not None:
            raise ValueError("The data could not be loaded")

    @staticmethod
    def objective(x_train_xgb, y_train_xgb, trial: optuna.trial.Trial) -> float:
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
        model = xgb.XGBRegressor(**param)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    @staticmethod
    def find_best_hyperparameters(x_train_xgb, y_train_xgb, n_trials: int = 100) -> dict:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: DiamondModel.objective(x_train_xgb, y_train_xgb, trial), n_trials=n_trials)
        return study.best_params

    @staticmethod
    def plot_gof(y_true: pd.Series, y_pred: pd.Series):
        plt.plot(y_true, y_pred, '.')
        plt.plot(y_true, y_true, linewidth=3, c='black')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()

    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        for extension, reader in file_readers.items():
            if file_path.endswith(extension):
                return reader(file_path)
        raise ValueError(f"Unsupported file extension: {file_path}")

    def _load(self, id: str):
        model_path = f'models/{id}_{self.model_name}.pkl'
        data_path = f'models/{id}_datas.csv'
        dummies_path = f'models/{id}_dummies.csv'
        processed_path = f'models/{id}_processed.csv'
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            self.model_trained = joblib.load(model_path)
            self.datas = pd.read_csv(data_path)
            self.datas_dummies = pd.read_csv(dummies_path)
            self.datas_processed = pd.read_csv(processed_path)
        else:
            raise ValueError("The model does not exist")

    def _save(self):
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model_trained, f'models/{self.id}_{self.model_name}.pkl')
        self.datas.to_csv(f'models/{self.id}_datas.csv', index=False)
        self.datas_dummies.to_csv(f'models/{self.id}_dummies.csv', index=False)
        self.datas_processed.to_csv(f'models/{self.id}_processed.csv', index=False)

    def clean_data(self, columns_to_drop: list[str] = ['depth', 'table', 'y', 'z'], columns_to_dummies: list[str] = ['cut', 'color', 'clarity']):
        self.datas = self.datas.dropna()
        self.datas = self.datas[(self.datas.x * self.datas.y * self.datas.z != 0) & (self.datas.price > 0)].reset_index(drop=True)
        self.datas_processed = self.datas.drop(columns=columns_to_drop)
        
        columns = [col for col in columns_to_dummies if col not in columns_to_drop]
        self.datas_dummies = pd.get_dummies(self.datas_processed, columns=columns, drop_first=True)

    def visualize_scatter_matrix(self, save: bool = False):
        fig = pd.plotting.scatter_matrix(self.datas.select_dtypes(include=['number']), figsize=(14, 10))
        if save:
            plt.savefig(f'visualizations/{self.id}_scatter_matrix.png')
        plt.show()
        return fig

    def visualize_histogram(self, save: bool = False):
        fig = self.datas.hist(bins=100, figsize=(14, 10))
        if save:
            plt.savefig(f'visualizations/{self.id}_histogram.png')
        plt.show()
        return fig

    def visualize_diamond_prices_by(self, cut_column: str, save: bool = False):
        fig = px.violin(self.datas, x=cut_column, y='price', color=cut_column, title=f'Price by {cut_column}')
        fig.show()
        if save:
            fig.write_html(f'visualizations/{self.id}_price_by_{cut_column}.html')
        return fig

    def train_model(self) -> float:
        if not hasattr(self, 'datas_dummies'):
            warnings.warn("The data is not yet cleaned, cleaning it now using the default columns to drop")
            self.clean_data()

        X = self.datas_dummies.drop(columns=['price'])
        y = self.datas_dummies['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_name == 'XGBRegressor':
            best_params = DiamondModel.find_best_hyperparameters(X_train, y_train, n_trials=100)
            self.model_trained = self.model(**best_params, enable_categorical=True, random_state=42)
        else:
            self.model_trained = self.model()

        if self.model_name == "LinearRegression":
            y_train = np.log(y_train)

        self.model_trained.fit(X_train, y_train)

        if self.model_name == "LinearRegression":
            y_pred = np.exp(self.model_trained.predict(X_test))
        else:
            y_pred = self.model_trained.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        self.predictions = y_pred
        self.GT_Y = y_test
        self._save()
        return mae

    def plot_predictions_vs_actual(self, save: bool = False):
        DiamondModel.plot_gof(self.GT_Y, self.predictions)
        if save:
            plt.savefig(f'visualizations/{self.id}_predictions_vs_actual.png')

    def predict(self, data: pd.DataFrame) -> pd.Series:
        predictions = self.model_trained.predict(data)
        if self.model_name == "LinearRegression":
            predictions = np.exp(predictions)
        return predictions

    def get_similar_samples(self, carat: float, cut: str, color: str, clarity: str, n: int) -> pd.DataFrame:
        if self.datas is None:
            raise ValueError("Data is not loaded.")
        if carat <= 0 or n <= 0:
            raise ValueError("Carat and n must be greater than 0")
        if carat > max(self.datas['carat']) or carat < min(self.datas['carat']):
            raise ValueError("Carat must be within the range of the dataset")
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

# You can now create an instance of DiamondModel and use it as needed.
