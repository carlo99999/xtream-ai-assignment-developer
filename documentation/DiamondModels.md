# Diamond Price Prediction Model Documentation

This documentation describes the `DiamondModel` class, which is designed to manage diamond price prediction models. The class provides functionality for data loading, cleaning, model training, evaluation, and prediction. The following sections provide detailed explanations of each part of the code and its usage.

## Dependencies

The script requires the following Python libraries:
- pandas
- sklearn
- joblib
- uuid
- xgboost
- matplotlib
- plotly
- warnings
- optuna
- logging
- numpy
- abc
- json
- torch
- torch.nn
- torch.optim

## Logging Configuration

```python
logging.basicConfig(level=logging.INFO)
```
This line configures the logging module to display messages at the INFO level and higher.

## BaseModel Class

The `BaseModel` class is an abstract base class that defines the interface for models. It includes methods for training, predicting, saving, loading, evaluating, and getting parameters.

```python
class BaseModel(ABC):
    ...
```

### Methods

- `train(X_train, y_train)`: Train the model.
  - **Description**: Trains the model with the provided training data.
  - **Params**:
    - **X_train**: Training features as a pandas DataFrame.
    - **y_train**: Training target as a pandas Series.
- `predict(X)`: Predict using the model.
  - **Description**: Makes predictions using the trained model.
  - **Params**:
    - **X**: Features to predict as a pandas DataFrame or Series.
  - **Returns**: Predictions as a pandas Series.
- `save(path)`: Save the model to a file.
  - **Description**: Saves the trained model to the specified path.
  - **Params**:
    - **path**: Path to save the model.
- `load(path)`: Load the model from a file.
  - **Description**: Loads the model from the specified path.
  - **Params**:
    - **path**: Path to load the model.
- `evaluate(X, y)`: Evaluate the model.
  - **Description**: Evaluates the model's performance on the given test data.
  - **Params**:
    - **X**: Features to evaluate as a pandas DataFrame.
    - **y**: True values as a pandas Series.
  - **Returns**: Evaluation metrics as a dictionary.
- `get_params()`: Get model parameters.
  - **Description**: Returns the parameters of the trained model.
  - **Returns**: Model parameters as a dictionary.

## LinearRegressionModel Class

The `LinearRegressionModel` class is a wrapper for the `LinearRegression` model from scikit-learn.

```python
class LinearRegressionModel(BaseModel):
    ...
```

### Methods

- `__init__(id=None)`: Initialize the model.
  - **Description**: Initializes the LinearRegression model. If an ID is provided, it loads the model with that ID.
  - **Params**:
    - **id**: Optional model ID to load a saved model.
- `train(X_train, y_train)`: Train the model.
  - **Description**: Trains the LinearRegression model with the provided training data.
  - **Params**:
    - **X_train**: Training features as a pandas DataFrame.
    - **y_train**: Training target as a pandas Series.
- `predict(X)`: Predict using the model.
  - **Description**: Makes predictions using the trained LinearRegression model.
  - **Params**:
    - **X**: Features to predict as a pandas DataFrame or Series.
  - **Returns**: Predictions as a pandas Series.
- `save(path)`: Save the model to a file.
  - **Description**: Saves the trained LinearRegression model to the specified path.
  - **Params**:
    - **path**: Path to save the model.
- `load(path)`: Load the model from a file.
  - **Description**: Loads the LinearRegression model from the specified path.
  - **Params**:
    - **path**: Path to load the model.
- `evaluate(X, y)`: Evaluate the model.
  - **Description**: Evaluates the LinearRegression model's performance on the given test data.
  - **Params**:
    - **X**: Features to evaluate as a pandas DataFrame.
    - **y**: True values as a pandas Series.
  - **Returns**: Evaluation metrics as a dictionary.
- `get_params()`: Get model parameters.
  - **Description**: Returns the parameters of the trained LinearRegression model.
  - **Returns**: Model parameters as a dictionary.

## MLP Class

The `MLP` class defines a Multi-Layer Perceptron (MLP) model using PyTorch.

```python
class MLP(nn.Module):
    ...
```

### Methods

- `__init__(input_dim, hidden_dim, output_dim, num_hidden_layers=5)`: Initialize the MLP model.
  - **Description**: Initializes the MLP model with the given dimensions and number of hidden layers.
  - **Params**:
    - **input_dim**: Dimension of the input features.
    - **hidden_dim**: Dimension of the hidden layers.
    - **output_dim**: Dimension of the output layer.
    - **num_hidden_layers**: Number of hidden layers in the model (default is 5).
- `forward(x)`: Forward pass through the MLP.
  - **Description**: Defines the forward pass for the MLP model.
  - **Params**:
    - **x**: Input tensor.
  - **Returns**: Output tensor.

## MLPModel Class

The `MLPModel` class is a wrapper for the MLP model using PyTorch.

```python
class MLPModel(BaseModel):
    ...
```

### Methods

- `__init__(input_dim=19, hidden_dim=128, output_dim=1, id=None)`: Initialize the MLP model.
  - **Description**: Initializes the MLP model with the given dimensions and optional ID.
  - **Params**:
    - **input_dim**: Number of input features (default is 19).
    - **hidden_dim**: Number of hidden units (default is 128).
    - **output_dim**: Number of output units (default is 1).
    - **id**: Optional model ID to load a saved model.
- `train(X_train, y_train, epochs=1000, lr=0.001)`: Train the model.
  - **Description**: Trains the MLP model with the provided training data, epochs, and learning rate.
  - **Params**:
    - **X_train**: Training features as a pandas DataFrame.
    - **y_train**: Training target as a pandas Series.
    - **epochs**: Number of training epochs (default is 1000).
    - **lr**: Learning rate (default is 0.001).
- `predict(X)`: Predict using the model.
  - **Description**: Makes predictions using the trained MLP model.
  - **Params**:
    - **X**: Features to predict as a pandas DataFrame or Series.
  - **Returns**: Predictions as a pandas Series.
- `save(path)`: Save the model to a file.
  - **Description**: Saves the trained MLP model to the specified path.
  - **Params**:
    - **path**: Path to save the model.
- `load(path)`: Load the model from a file.
  - **Description**: Loads the MLP model from the specified path.
  - **Params**:
    - **path**: Path to load the model.
- `evaluate(X, y)`: Evaluate the model.
  - **Description**: Evaluates the MLP model's performance on the given test data.
  - **Params**:
    - **X**: Features to evaluate as a pandas DataFrame.
    - **y**: True values as a pandas Series.
  - **Returns**: Evaluation metrics as a dictionary.
- `get_params()`: Get model parameters.
  - **Description**: Returns the parameters of the trained MLP model.
  - **Returns**: Model parameters as a dictionary.

## XGBRegressorModel Class

The `XGBRegressorModel` class is a wrapper for the `XGBRegressor` model from xgboost.

```python
class XGBRegressorModel(BaseModel):
    ...
```

### Methods

- `__init__(id=None)`: Initialize the model.
  - **Description**: Initializes the XGBRegressor model. If an ID is provided, it loads the model with that ID.
  - **Params**:
    - **id**: Optional model ID to load a saved model.
- `train(X_train, y_train)`: Train the model.
  - **Description**: Trains the XGBRegressor model with the provided training data.
  - **Params**:
    - **X_train**: Training features as a pandas DataFrame.
    - **y_train**: Training target as a pandas Series.
- `predict(X)`: Predict using the model.
  - **Description**: Makes predictions using the trained XGBRegressor model.
  - **Params**:
    - **X**: Features to predict as a pandas DataFrame or Series.
  - **Returns**: Predictions as a pandas Series.
- `save(path)`: Save the model to a file.
  - **Description**: Saves the trained XGBRegressor model to the specified path.
  - **Params**:
    - **path**: Path to save

 the model.
- `load(path)`: Load the model from a file.
  - **Description**: Loads the XGBRegressor model from the specified path.
  - **Params**:
    - **path**: Path to load the model.
- `evaluate(X_test, y)`: Evaluate the model.
  - **Description**: Evaluates the XGBRegressor model's performance on the given test data.
  - **Params**:
    - **X_test**: Features to evaluate as a pandas DataFrame.
    - **y**: True values as a pandas Series.
  - **Returns**: Evaluation metrics as a dictionary.
- `get_params()`: Get model parameters.
  - **Description**: Returns the parameters of the trained XGBRegressor model.
  - **Returns**: Model parameters as a dictionary.

### Static Methods

- `objective(x_train_xgb, y_train_xgb, trial)`: Objective function for hyperparameter optimization using Optuna.
  - **Description**: Defines the objective function for Optuna hyperparameter optimization.
  - **Params**:
    - **x_train_xgb**: Training features for XGBoost as a pandas DataFrame.
    - **y_train_xgb**: Training labels for XGBoost as a pandas Series.
    - **trial**: Optuna trial object.
  - **Returns**: Mean absolute error on validation set.
- `find_best_hyperparameters(x_train_xgb, y_train_xgb, n_trials=100)`: Find the best hyperparameters for XGBoost using Optuna.
  - **Description**: Uses Optuna to find the best hyperparameters for the XGBRegressor model.
  - **Params**:
    - **x_train_xgb**: Training features for XGBoost as a pandas DataFrame.
    - **y_train_xgb**: Training labels for XGBoost as a pandas Series.
    - **n_trials**: Number of trials for Optuna optimization (default is 100).
  - **Returns**: Best hyperparameters found as a dictionary.

## File Readers

Dictionary of functions to read various file types.

```python
file_readers = {
    '.csv': pd.read_csv,
    '.xlsx': pd.read_excel,
    ...
}
```

## Modelling Algorithms

Dictionary of available modelling algorithms.

```python
modelling_algorithms = {
    "XGBRegressor": XGBRegressorModel,
    "LinearRegression": LinearRegressionModel,
    "MLP": MLPModel
}
```

## DiamondModel Class

The `DiamondModel` class is designed to manage diamond price prediction models. It provides functionality for data loading, cleaning, model training, evaluation, and prediction.

```python
class DiamondModel:
    ...
```

### Methods

- `__init__(datas=None, id=None, model=None, folder="models")`: Initialize the DiamondModel.
  - **Description**: Initializes the DiamondModel with the provided data, model, and optional ID.
  - **Params**:
    - **datas**: The data to train the model, either a pandas DataFrame or a path to a file.
    - **id**: The ID of the model to load.
    - **model**: The model to train the data. Options: 'LinearRegression', 'XGBRegressor'.
    - **folder**: The folder where models and data are saved/loaded (default is "models").

- `_check_if_model_exists(id)`: Check if a model with the given ID already exists.
  - **Description**: Checks if a model with the specified ID already exists in the folder.
  - **Params**:
    - **id**: Model ID.
  - **Returns**: True if the model exists, False otherwise.

- `_create_new_model(datas, folder, model, id=None)`: Create a new model.
  - **Description**: Creates a new model with the provided data, folder, and model type.
  - **Params**:
    - **datas**: The data to train the model.
    - **folder**: The folder where models and data are saved/loaded.
    - **model**: The model to train the data.
    - **id**: Optional model ID.

- `plot_gof(y_true, y_pred, show)`: Plot the goodness of fit.
  - **Description**: Plots the goodness of fit between actual and predicted values.
  - **Params**:
    - **y_true**: Actual values as a pandas Series.
    - **y_pred**: Predicted values as a pandas Series.
    - **show**: Whether to display the plot.

- `prepare_data_for_prediction(data_blueprint, data_to_predict, target_column="price")`: Prepare data for prediction.
  - **Description**: Prepares the data to be predicted by ensuring it has the same columns as the training data blueprint.
  - **Params**:
    - **data_blueprint**: The dataframe blueprint with the expected columns.
    - **data_to_predict**: The new data to prepare for prediction.
    - **target_column**: The target column to exclude from prediction data (default is "price").
  - **Returns**: The prepared dataframe with the same columns as the blueprint.

- `_load_data_from_file(file_path)`: Load data from a file.
  - **Description**: Loads data from the specified file path using the appropriate reader function.
  - **Params**:
    - **file_path**: Path to the file.
  - **Returns**: Loaded data as a pandas DataFrame.

- `_load(id)`: Load a model and its data from files.
  - **Description**: Loads the model and its associated data from the folder using the provided ID.
  - **Params**:
    - **id**: Model ID.

- `_validate_loaded_data()`: Validate that the required data and model components are loaded.
  - **Description**: Validates that the necessary data and model components are properly loaded.
  - Raises a ValueError if any component is missing.

- `_save(folder_to_save=None)`: Save the model and its data to files.
  - **Description**: Saves the model and its associated data to the specified folder.
  - **Params**:
    - **folder_to_save**: Folder where to save the files (default is None).

- `clean_data(columns_to_drop=['depth', 'table', 'y', 'z'], columns_to_dummies=['cut', 'color', 'clarity'])`: Clean the data.
  - **Description**: Cleans the data by dropping specified columns and creating dummy variables for categorical features.
  - **Params**:
    - **columns_to_drop**: List of columns to drop (default is ['depth', 'table', 'y', 'z']).
    - **columns_to_dummies**: List of columns to convert to dummy variables (default is ['cut', 'color', 'clarity']).

- `visualize_scatter_matrix(save=False, show=True, path='visualizations')`: Visualize a scatter matrix of the numerical features.
  - **Description**: Visualizes a scatter matrix of the numerical features in the data.
  - **Params**:
    - **save**: Whether to save the plot to a file (default is False).
    - **show**: Whether to display the plot (default is True).
    - **path**: Path to save the plot (default is 'visualizations').
  - **Returns**: The scatter matrix figure.

- `visualize_histogram(save=False, show=True, path='visualizations')`: Visualize histograms of the features.
  - **Description**: Visualizes histograms of the features in the data.
  - **Params**:
    - **save**: Whether to save the plot to a file (default is False).
    - **show**: Whether to display the plot (default is True).
    - **path**: Path to save the plot (default is 'visualizations').
  - **Returns**: The histogram figure.

- `visualize_diamond_prices_by(cut_column, save=False, show=True, path='visualizations')`: Visualize diamond prices by a categorical feature using a violin plot.
  - **Description**: Visualizes diamond prices by a specified categorical feature using a violin plot.
  - **Params**:
    - **cut_column**: The categorical column to plot.
    - **save**: Whether to save the plot to a file (default is False).
    - **show**: Whether to display the plot (default is True).
    - **path**: Path to save the plot (default is 'visualizations').
  - **Returns**: The violin plot figure.

- `train_model(folder_to_save=None)`: Train the model and save it to a file.
  - **Description**: Trains the model with the cleaned data and saves the trained model and data to a specified folder.
  - **Params**:
    - **folder_to_save**: Folder where to save the trained model and data (default is None).
  - **Returns**: Mean absolute error and mean squared error on the test set.

- `plot_predictions_vs_actual(save=False, show=False, path='visualization')`: Plot the predicted values against the actual values.
  - **Description**: Plots the predicted values against the actual values for evaluation.
  - **Params**:
    - **save**: Whether to save the plot to a file (default is False).
    - **show**: Whether to display the plot (default is False).
    - **path**: Path to save the plot (default is 'visualization').

- `predict(data)`: Predict prices for new data.
  - **Description**: Uses the trained model to predict prices for new data.


  - **Params**:
    - **data**: DataFrame containing the features for prediction.
  - **Returns**: Predicted prices as a pandas Series.

- `get_similar_samples(carat, cut, color, clarity, n)`: Get similar diamond samples based on carat, cut, color, and clarity.
  - **Description**: Retrieves similar diamond samples from the dataset based on specified carat, cut, color, and clarity.
  - **Params**:
    - **carat**: Carat weight of the diamond.
    - **cut**: Cut of the diamond.
    - **color**: Color of the diamond.
    - **clarity**: Clarity of the diamond.
    - **n**: Number of similar samples to return.
  - **Returns**: Similar samples as a pandas DataFrame.

## Usage Example

```python
# Create a new DiamondModel
diamond_model = DiamondModel(datas="path_to_csv_file.csv", model="LinearRegression")

# Clean the data
diamond_model.clean_data()

# Train the model
metrics = diamond_model.train_model()

# Predict prices for new data
new_data = pd.DataFrame({...})
predictions = diamond_model.predict(new_data)

# Visualize results
diamond_model.visualize_scatter_matrix()
diamond_model.visualize_histogram()
diamond_model.visualize_diamond_prices_by("cut")
```

## Conclusion

This documentation provides an overview of the `DiamondModel` class and its associated components. The class is designed to handle various aspects of diamond price prediction, including data loading, cleaning, model training, evaluation, and visualization. The provided methods and their detailed descriptions ensure that users can effectively utilize the class for their predictive modeling needs.