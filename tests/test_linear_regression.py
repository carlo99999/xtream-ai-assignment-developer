import pytest
import pandas as pd
import numpy as np
from DiamondModels import LinearRegressionModel

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [3, 6, 9, 12, 15]
    })

@pytest.fixture
def linear_model():
    return LinearRegressionModel()

def test_linear_regression_initialization(linear_model):
    assert linear_model.model is not None

def test_linear_regression_train(linear_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    linear_model.train(X, y)
    assert linear_model.model_trained is not None

def test_linear_regression_predict(linear_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    linear_model.train(X, y)
    predictions = linear_model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, pd.Series)

def test_linear_regression_evaluate(linear_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    linear_model.train(X, y)
    metrics = linear_model.evaluate(X, y)
    assert "mae" in metrics
    assert "mse" in metrics

def test_linear_regression_get_params(linear_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    linear_model.train(X, y)
    params = linear_model.get_params()
    assert "fit_intercept" in params
    assert "copy_X" in params
    assert "n_jobs" in params

def test_linear_regression_save_load(linear_model, sample_data, tmp_path):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    linear_model.train(X, y)
    
    save_path = tmp_path / "linear_model.pkl"
    linear_model.save(str(save_path))
    
    loaded_model = LinearRegressionModel()
    loaded_model.load(str(save_path))
    
    assert loaded_model.model_trained is not None
    assert np.allclose(loaded_model.predict(X), linear_model.predict(X))