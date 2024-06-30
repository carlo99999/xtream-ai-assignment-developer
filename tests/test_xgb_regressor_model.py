import pytest
import pandas as pd
import numpy as np
from DiamondModels import XGBRegressorModel

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [3, 6, 9, 12, 15]
    })

@pytest.fixture
def xgb_model():
    return XGBRegressorModel()

def test_xgb_regressor_initialization(xgb_model):
    assert xgb_model.model is not None

def test_xgb_regressor_train(xgb_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    xgb_model.train(X, y)
    assert xgb_model.model_trained is not None

def test_xgb_regressor_predict(xgb_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    xgb_model.train(X, y)
    predictions = xgb_model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, pd.Series)

def test_xgb_regressor_evaluate(xgb_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    xgb_model.train(X, y)
    metrics = xgb_model.evaluate(X, y)
    assert "mae" in metrics
    assert "mse" in metrics

def test_xgb_regressor_get_params(xgb_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    xgb_model.train(X, y)
    params = xgb_model.get_params()
    assert "n_estimators" in params

def test_xgb_regressor_save_load(xgb_model, sample_data, tmp_path):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    xgb_model.train(X, y)
    
    save_path = tmp_path / "xgb_model.pkl"
    xgb_model.save(str(save_path))
    
    loaded_model = XGBRegressorModel()
    loaded_model.load(str(save_path))
    
    assert loaded_model.model_trained is not None
    assert np.allclose(loaded_model.predict(X), xgb_model.predict(X))

def test_xgb_regressor_find_best_hyperparameters(xgb_model, sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    best_params = XGBRegressorModel.find_best_hyperparameters(X, y, n_trials=10)
    assert isinstance(best_params, dict)
    assert "lambda" in best_params
    assert "alpha" in best_params
    assert "learning_rate" in best_params