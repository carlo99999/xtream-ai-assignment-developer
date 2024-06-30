import pytest
from abc import ABC
from DiamondModels import BaseModel
import pandas as pd
import numpy as np

class ConcreteBaseModel(BaseModel):
    def train(self, X_train, y_train):
        pass
    
    def predict(self, X):
        return pd.Series(np.random.rand(len(X)))
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def evaluate(self, X, y):
        return {"mae": 0.1, "mse": 0.01}
    
    def get_params(self):
        return {"param1": 1, "param2": 2}

@pytest.fixture
def base_model():
    return ConcreteBaseModel()

def test_base_model_initialization(base_model):
    assert base_model.model_trained is None
    assert base_model.mae_mse is None
    assert base_model.params is None

def test_base_model_calculate_metrics(base_model):
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, 2.1, 2.9, 4.2, 5.1])
    metrics = base_model._calculate_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "mse" in metrics
    assert metrics["mae"] > 0
    assert metrics["mse"] > 0

def test_base_model_abstract_methods():
    with pytest.raises(TypeError):
        BaseModel()