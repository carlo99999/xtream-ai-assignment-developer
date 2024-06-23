# Diamond Price Prediction Model Documentation

This documentation provides an overview and explanation of the code used for creating and managing a diamond price prediction model. The model can be trained on a dataset, and existing models can be loaded for predictions. The code utilizes various libraries and techniques to achieve these tasks.

## Overview

The code is structured to handle data loading, model training, hyperparameter tuning, and prediction tasks. It uses several key libraries to perform these tasks efficiently.

## Imports and Setup

The necessary libraries are imported at the beginning of the script. Each library serves a specific purpose in the model's functionality.

```python
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
```

### Explanation of Imports
- **Pandas (`pd`)**: For data manipulation and analysis.
- **Scikit-learn**: For model training and evaluation.
  - `train_test_split`: Splits the dataset into training and testing sets.
  - `mean_absolute_error`: Evaluates the model's performance.
  - `LinearRegression`: Linear regression model.
- **Joblib (`joblib`)**: For saving and loading models.
- **UUID (`uuid`)**: For generating unique model IDs.
- **XGBoost (`xgb`)**: For gradient boosting algorithms.
  - `XGBRegressor`: XGBoost regression model.
- **OS (`os`)**: For interacting with the operating system.
- **Matplotlib (`plt`)**: For plotting graphs.
- **Plotly (`px`)**: For interactive visualizations.
- **Warnings (`warnings`)**: For handling warnings.
- **Optuna (`optuna`)**: For hyperparameter optimization.
- **Typing (`Union`)**: For type annotations.
- **Numpy (`np`)**: For numerical operations.

## File Readers

A dictionary to map file extensions to their respective pandas file reading functions. This allows for flexible data loading from various file formats.

```python
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
```

### Explanation
- Maps file extensions to the corresponding pandas function for reading that file type.

## Modelling Algorithms

A dictionary to map model names to their respective classes.

```python
modelling_algorithms = {
    "XGBRegressor": XGBRegressor,
    "LinearRegression": LinearRegression,
}
```

### Explanation
- Allows easy selection and initialization of different modeling algorithms.

## DiamondModel Class

The `DiamondModel` class encapsulates the entire workflow for data handling, model training, and prediction. 

### Initialization

The constructor initializes the model either with data for training or an ID for loading an existing model.

```python
class DiamondModel:
    def __init__(self, datas: Union[pd.DataFrame, str] = None, id: str = None, model: str = None, folder: str = "models") -> None:
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
        
        if id and datas is None:
            self._load(id, folder)
        
        if self.datas is None and datas is not None:
            raise ValueError("The data could not be loaded")
```

### Explanation
- Initializes the model with data or loads an existing model by its ID.
- Checks for valid inputs and raises appropriate errors if necessary.

### Hyperparameter Optimization

The class includes methods for optimizing hyperparameters using Optuna.

```python
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
```

### Explanation
- `objective`: Defines the objective function for the Optuna study, which tunes hyperparameters to minimize mean absolute error (MAE).
- `find_best_hyperparameters`: Conducts the hyperparameter optimization process using the defined objective function.

### Data Handling Methods

The class includes methods to load, save, and clean data.

#### Loading Data

```python
def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
    for extension, reader in file_readers.items():
        if file_path.endswith(extension):
            return reader(file_path)
    raise ValueError(f"Unsupported file extension: {file_path}")
```

### Explanation
- Loads data from a file based on its extension.

#### Saving and Loading Models

```python
def _load(self, id: str, folder: str = 'models'):
    model_path = f'{folder}/{id}_{self.model_name}.pkl'
    data_path = f'{folder}/{id}_datas.csv'
    dummies_path = f'{folder}/{id}_dummies.csv'
    processed_path = f'{folder}/{id}_processed.csv'
    if os.path.exists(model_path) and os.path.exists(data_path):
        self.model_trained = joblib.load(model_path)
        self.datas = pd.read_csv(data_path)
        self.datas_dummies = pd.read_csv(dummies_path)
        self.datas_processed = pd.read_csv(processed_path)
    else:
        raise ValueError("The model does not exist")

def _save(self, folder_to_save: str = None):
    folder_to_save = folder_to_save if folder_to_save else 'models'
    os.makedirs(folder_to_save, exist_ok=True)
    joblib.dump(self.model_trained, f'{folder_to_save}/{self.id}_{self.model_name}.pkl')
    self.datas.to_csv(f'{folder_to_save}/{self.id}_datas.csv', index=False)
    self.datas_dummies.to_csv(f'{folder_to_save}/{self.id}_dummies.csv', index=False)
    self.datas_processed.to_csv(f'{folder_to_save}/{self.id}_processed.csv', index=False)
```

### Explanation
- `_load`: Loads an existing model and its associated data from disk.
- `_save`: Saves the trained model and its data to disk.

#### Cleaning Data

```python
def clean_data(self, columns_to_drop: list[str] = ['depth', 'table', 'y', 'z'], columns_to_dummies: list[str] = ['cut', 'color', 'clarity']):
    self.datas = self.datas.dropna()
    self.datas = self.datas[(self.datas.x * self.datas.y * self.datas.z != 0) & (self.datas.price > 0)].reset_index(drop=True

)
    self.datas_processed = self.datas.drop(columns=columns_to_drop)
    
    columns = [col for col in columns_to_dummies if col not in columns_to_drop]
    self.datas_dummies = pd.get_dummies(self.datas_processed, columns=columns, drop_first=True)
```

### Explanation
- `clean_data`: Cleans the data by dropping specified columns, removing rows with invalid values, and encoding categorical variables.

### Visualization Methods

The class includes methods to visualize data and model predictions.

#### Scatter Matrix

```python
def visualize_scatter_matrix(self, save: bool = False):
    fig = pd.plotting.scatter_matrix(self.datas.select_dtypes(include=['number']), figsize=(14, 10))
    if save:
        plt.savefig(f'visualizations/{self.id}_scatter_matrix.png')
    plt.show()
    return fig
```

### Explanation
- `visualize_scatter_matrix`: Creates and displays a scatter matrix of numerical features in the dataset.

#### Histogram

```python
def visualize_histogram(self, save: bool = False):
    fig = self.datas.hist(bins=100, figsize=(14, 10))
    if save:
        plt.savefig(f'visualizations/{self.id}_histogram.png')
    plt.show()
    return fig
```

### Explanation
- `visualize_histogram`: Creates and displays histograms for each feature in the dataset.

#### Violin Plot

```python
def visualize_diamond_prices_by(self, cut_column: str, save: bool = False):
    fig = px.violin(self.datas, x=cut_column, y='price', color=cut_column, title=f'Price by {cut_column}')
    fig.show()
    if save:
        fig.write_html(f'visualizations/{self.id}_price_by_{cut_column}.html')
    return fig
```

### Explanation
- `visualize_diamond_prices_by`: Creates and displays a violin plot showing diamond prices by a specified categorical feature.

### Model Training and Prediction

#### Training the Model

```python
def train_model(self, folder_to_save: str = None) -> float:
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
    self._save(folder_to_save)
    return mae
```

### Explanation
- `train_model`: Trains the model on the cleaned dataset, optionally optimizing hyperparameters for XGBoost. It calculates the mean absolute error (MAE) and saves the model.

#### Making Predictions

```python
def predict(self, data: pd.DataFrame) -> pd.Series:
    predictions = self.model_trained.predict(data)
    if self.model_name == "LinearRegression":
        predictions = np.exp(predictions)
    return predictions
```

### Explanation
- `predict`: Uses the trained model to make predictions on new data, applying the exponential function for linear regression predictions.

### Visualization of Predictions

```python
def plot_predictions_vs_actual(self, save: bool = False):
    DiamondModel.plot_gof(self.GT_Y, self.predictions)
    if save:
        plt.savefig(f'visualizations/{self.id}_predictions_vs_actual.png')
```

### Explanation
- `plot_predictions_vs_actual`: Plots the predictions against actual values to evaluate the model's performance.

### Finding Similar Samples

```python
def get_similar_samples(self, carat: float, cut: str, color: str, clarity: str, n: int) -> pd.DataFrame:
    if self.datas is None:
        raise ValueError("Data is not loaded.")
    if carat <= 0 or n <= 0:
        raise ValueError("Carat and n must be greater than 0")
    if carat > max(self.datas['carat']) * 1.2 or carat < min(self.datas['carat']) * 0.8:
        raise ValueError("Carat must be within the range of the dataset")
    if n > len(self.datas):
        return self.datas
    if clarity not in self.datas['clarity'].unique() or color not in self.datas['color'].unique() or cut not in self.datas['cut'].unique():
        raise ValueError("Cut, color, and clarity must be one of the unique values in the dataset")

    filtered_data = self.datas[(self.datas['cut'] == cut) & (self.datas['color'] == color) & (self.datas['clarity'] == clarity)]

    if filtered_data.empty:
        return pd.DataFrame()
    
    filtered_data['weight_diff'] = (filtered_data['carat'] - carat).abs()
    similar_samples = filtered_data.sort_values(by='weight_diff')
    
    print(similar_samples.shape)
    return similar_samples.drop(columns=['weight_diff'])
```

### Explanation
- `get_similar_samples`: Finds samples in the dataset similar to specified attributes, useful for comparative analysis.

## Conclusion

This documentation provides a detailed overview of the Diamond Price Prediction Model, highlighting its key functionalities and explaining the underlying code. The class `DiamondModel` encapsulates data handling, model training, hyperparameter optimization, and prediction functionalities, making it a comprehensive solution for diamond price prediction tasks.