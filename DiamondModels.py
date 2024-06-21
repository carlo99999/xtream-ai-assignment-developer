import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import joblib
import uuid
import xgboost
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import optuna
from typing import Union

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
    def __init__(self, datas: Union[pd.DataFrame, str]=None, id: str=None, model: str=None) -> None:
        self.datas = None
        self.model = None
        self.modelling_algorithms = {
            "LinearRegression": LinearRegression,
            "XGBRegressor": XGBRegressor
        }
        self.id = uuid.uuid4().hex
        self.model_name = model

        if datas is None and id is None:
            raise ValueError("You must provide data to train the model or an id to load the model")
        if id is None and model is None:
            raise ValueError("You must decide a model to train the data")
        if model and model not in self.modelling_algorithms:
            raise ValueError("The model provided is not yet supported")
        
        if isinstance(datas, str):
            self.datas = self._load_data_from_file(datas)
        elif isinstance(datas, pd.DataFrame):
            self.datas = datas

        if id:
            self._load(id)
        
        if self.datas is None:
            raise ValueError("The data could not be loaded")

        if model:
            self.model = self.modelling_algorithms[model]()

    @staticmethod
    def objective(x_train_xbg, y_train_xbg, trial: optuna.trial.Trial) -> float:
        # Define hyperparameters to tune
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

        # Split the training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train_xbg, y_train_xbg, test_size=0.2, random_state=42)

        # Train the model
        model = xgboost.XGBRegressor(**param)
        model.fit(x_train, y_train)

        # Make predictions
        preds = model.predict(x_val)

        # Calculate MAE
        mae = mean_absolute_error(y_val, preds)

        return mae

    @staticmethod
    def find_best_hyperparameters(x_train_xbg, y_train_xbg, n_trials: int=100) -> dict:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: DiamondModel.objective(x_train_xbg, y_train_xbg, trial), n_trials=n_trials)
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
        model_path = f'models/{id}.pkl'
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise ValueError("The model does not exist")

    def _save(self):
        joblib.dump(self.model, f'models/{self.id}.pkl')

    def clean_data(self, columns_to_drop: list[str] = ['depth', 'table', 'y', 'z']):
        """
        Clean the data by dropping unnecessary columns and rows with missing values.
        Params:
        columns_to_drop: list[str]: The columns to drop from the data.
        """
        self.datas = self.datas.dropna()
        self.datas = self.datas[(self.datas.x * self.datas.y * self.datas.z != 0) & (self.datas.price > 0)]
        self.datas = self.datas.reset_index(drop=True)
        self.datas_processed = self.datas.drop(columns=columns_to_drop)
        self.datas_dummies = pd.get_dummies(self.datas_processed)

    def visualize_scatter_matrix(self, save: bool=False) -> None:
        """
        Visualize the correlation between numeric columns in the data.
        """
        pd.plotting.scatter_matrix(self.datas.select_dtypes(include=['number']), figsize=(14, 10))
        if save:
            plt.savefig(f'visualizations/{self.id}_scatter_matrix.png')
        plt.show()

    def visualize_histogram(self, save: bool=False) -> None:
        """
        Visualize the histogram of the data.
        save : bool: Whether to save the plot or not.
        """
        self.datas.hist(bins=100, figsize=(14, 10))
        if save:
            plt.savefig(f'visualizations/{self.id}_histogram.png')
        plt.show()

    def visualize_diamond_prices_by(self, cut_column: str,save: bool=False) -> None:
        """
        Visualize the prices of diamonds by a specific column.
        Params:
        cut_column: str: The column to visualize the prices by.
        """
        fig = px.violin(self.datas, x=cut_column, y='price', color=cut_column, title=f'Price by {cut_column}')
        fig.show()
        if save:
            fig.write_html(f'visualizations/{self.id}_price_by_{cut_column}.html')

    def train_model(self) -> float:
        """
        Train the model and return the mean absolute error (MAE).
        """
        if not hasattr(self, 'datas_dummies'):
            warnings.warn("The data is not yet cleaned, cleaning it now using the default columns to drop")
            self.clean_data()

        X = self.datas_dummies.drop(columns=['price'])
        y = self.datas_dummies['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_name == 'XGBRegressor':
            best_params = DiamondModel.find_best_hyperparameters(X_train, y_train)
            self.model = self.model(**best_params, enable_categorical=True, random_state=42)
        else:
            self.model = self.model()

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        self.predictions = y_pred
        self.GT_Y = y_test
        self._save()
        return mae

    def plot_predictions_vs_actual(self, save: bool=False) -> None:
        """
        Plot the predictions vs the actual values.
        save: bool: Whether to save the plot or not.
        """
        DiamondModel.plot_gof(self.GT_Y, self.predictions)
        if save:
            plt.savefig(f'visualizations/{self.id}_predictions_vs_actual.png')
        
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the prices of diamonds using the trained model.
        Params:
        data: pd.DataFrame: The data to predict the prices of diamonds.
        """
        return self.model.predict(data)
    
    def get_similar_samples(self, carat: float, cut: str, color: str, clarity: str, n: int) -> pd.DataFrame:
        """
        Get similar samples to a given diamond.
        Params:
        carat: float: The carat of the diamond.
        cut: str: The cut of the diamond.
        color: str: The color of the diamond.
        clarity: str: The clarity of the diamond.
        n: int: The number of similar samples to return.
        """
        if self.datas is None:
            raise ValueError("Data is not loaded.")
        if carat <= 0 or n <= 0:
            raise ValueError("Carat and n must be greater than 0")
        if carat>max(self.datas['carat']) or carat<min(self.datas['carat']):
            raise ValueError("Carat must be within the range of the dataset")
        if n>len(self.datas):
            return self.datas
        if clarity not in self.datas['clarity'].unique():
            raise ValueError("Clarity must be one of the unique values in the dataset")
        if color not in self.datas['color'].unique():
            raise ValueError("Color must be one of the unique values in the dataset")
        if cut not in self.datas['cut'].unique():
            raise ValueError("Cut must be one of the unique values in the dataset")
        
        
        # Filtra il dataset
        filtered_data = self.datas[
            (self.datas['cut'] == cut) &
            (self.datas['color'] == color) &
            (self.datas['clarity'] == clarity)
        ]
        
        if filtered_data.empty:
            return pd.DataFrame()  # Restituisce un DataFrame vuoto se non ci sono campioni simili

        # Calcola la differenza di peso e ordina per similarità di peso
        filtered_data['weight_diff'] = (filtered_data['carat'] - carat).abs()
        similar_samples = filtered_data.sort_values(by='weight_diff').head(n)
        
        # Rimuovi la colonna temporanea 'weight_diff'
        similar_samples = similar_samples.drop(columns=['weight_diff'])
        
        return similar_samples

        
        
    
