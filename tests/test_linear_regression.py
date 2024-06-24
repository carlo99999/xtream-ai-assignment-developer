import pytest
import pandas as pd
import numpy as np
from DiamondModels import LinearRegressionModel

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.rand(100), name='target')
    return X, y

def test_train(sample_data):
    X, y = sample_data
    model = LinearRegressionModel()
    model.train(X, y)
    assert model.model is not None

def test_predict(sample_data):
    X, y = sample_data
    model = LinearRegressionModel()
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)

def test_save_load_model(sample_data, tmp_path):
    X, y = sample_data
    model = LinearRegressionModel()
    model.train(X, y)
    model_path = tmp_path / "linear_model.pkl"
    model.save(model_path)
    assert model_path.exists()

    new_model = LinearRegressionModel()
    new_model.load(model_path)
    new_predictions = new_model.predict(X)
    assert np.allclose(model.predict(X), new_predictions)
