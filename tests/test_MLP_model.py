import pytest
import pandas as pd
import numpy as np
import torch
from DiamondModels import MLPModel

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [3, 6, 9, 12, 15]
    })

@pytest.fixture
def mlp_model():
    return MLPModel(input_dim=2)

def test_mlp_initialization(mlp_model):
    assert mlp_model.model is not None
    assert isinstance(mlp_model.model, torch.nn.Module)

def test_mlp_train(mlp_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    mlp_model.train(X, y, epochs=10)
    assert mlp_model.model_trained is not None

def test_mlp_predict(mlp_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    mlp_model.train(X, y, epochs=10)
    predictions = mlp_model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, pd.Series)

def test_mlp_evaluate(mlp_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    mlp_model.train(X, y, epochs=10)
    metrics = mlp_model.evaluate(X, y)
    assert "mae" in metrics
    assert "mse" in metrics

def test_mlp_get_params(mlp_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    mlp_model.train(X, y, epochs=10)
    params = mlp_model.get_params()
    assert "hidden_units" in params
    assert "learning_rate" in params

def test_mlp_save_load(mlp_model, sample_data, tmp_path):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    mlp_model.train(X, y, epochs=10)
    
    save_path = tmp_path / "mlp_model.pkl"
    mlp_model.save(str(save_path))
    
    loaded_model = MLPModel(input_dim=2)
    loaded_model.load(str(save_path))
    
    assert loaded_model.model_trained is not None
    assert np.allclose(loaded_model.predict(X).values, mlp_model.predict(X).values, atol=1e-4)

def test_mlp_scaler(mlp_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    mlp_model.train(X, y, epochs=10)
    assert mlp_model.scaler is not None

    # Test if scaler is correctly applied during prediction
    scaled_predictions = mlp_model.predict(X)
    assert not np.allclose(scaled_predictions.values, y.values)  # Predictions should be scaled