# ModelsPipeline Documentation

This documentation provides a detailed explanation of the `ModelsPipeline` class, which is designed to streamline the process of training multiple machine learning models on a given dataset. The class uses the `DiamondModel` and various algorithms to automate data loading, cleaning, training, evaluation, and visualization.

## Dependencies

The script requires the following Python libraries:
- pandas
- os

Additionally, it imports the following from the `DiamondModels` module:
- `DiamondModel`
- `XGBRegressorModel`
- `LinearRegressionModel`
- `MLPModel`
- `file_readers`
- `modelling_algorithms`

## ModelsPipeline Class

The `ModelsPipeline` class handles the end-to-end process of loading data, training different models, and saving the results along with visualizations.

```python
class ModelsPipeline:
    ...
```

### Methods

- `__init__(data_path: str, folder: str = "ModelsPipeline")`
  - **Description**: Initializes the ModelsPipeline with the path to the data and the folder where models and visualizations will be saved.
  - **Params**:
    - **data_path**: Path to the dataset file.
    - **folder**: Folder where models and visualizations will be saved (default is "ModelsPipeline").

- `run_pipeline()`
  - **Description**: Runs the entire pipeline, including loading data, training models, and saving visualizations.
  - **Steps**:
    1. Load data from the specified path.
    2. Iterate over the defined models and train each one.
    3. Save the trained model and generate visualizations for each model.

- `_load_data(data_path: str) -> pd.DataFrame`
  - **Description**: Loads data from the specified file path using the appropriate reader function based on the file extension.
  - **Params**:
    - **data_path**: Path to the dataset file.
  - **Returns**: Loaded data as a pandas DataFrame.
  - **Raises**: ValueError if the file extension is not supported.

## Example Usage

The following example demonstrates how to use the `ModelsPipeline` class to train models on a dataset of diamond prices.

```python
# Example usage
data_path = "data/diamonds.csv"
pipeline = ModelsPipeline(data_path)
pipeline.run_pipeline()
```

## Detailed Method Descriptions

### `__init__(data_path: str, folder: str = "ModelsPipeline")`

This method initializes the `ModelsPipeline` instance with the provided data path and folder for saving outputs. It creates the necessary directory structure.

### `run_pipeline()`

This method orchestrates the entire process:
1. Loads the dataset from the specified path.
2. Iterates over the models defined in `modelling_algorithms`.
3. For each model:
   - Initializes a `DiamondModel` instance with the dataset and model type.
   - Cleans the dataset.
   - Trains the model and saves it along with the performance metrics.
   - Generates and saves various visualizations (scatter matrix, histograms, violin plots, and prediction vs. actual plots).
4. Prints the performance metrics (MAE and MSE) for each trained model.

### `_load_data(data_path: str) -> pd.DataFrame`

This helper method loads data from a file using the appropriate reader function based on the file extension. It removes any columns with 'Unnamed' in their name.

```python
def _load_data(self, data_path: str) -> pd.DataFrame:
    for extension, reader in file_readers.items():
        if data_path.endswith(extension):
            df = reader(data_path)
            for i in df.columns:
                if i.find('Unnamed') != -1:
                    df.drop(columns=[i], inplace=True)
            return df
    raise ValueError(f"Unsupported file extension: {data_path}")
```

## Conclusion

The `ModelsPipeline` class provides an efficient way to automate the training and evaluation of multiple machine learning models on a dataset. It simplifies the process by handling data loading, cleaning, training, evaluation, and visualization in a systematic manner. By following this documentation, users can effectively utilize the class for their predictive modeling needs.