import pytest
import pandas as pd
from DiamondModels import DiamondModel

@pytest.fixture
def diamond_data():
    data = {
        'carat': [0.23, 0.21, 0.23, 0.29, 0.31],
        'cut': ['Ideal', 'Premium', 'Good', 'Premium', 'Good'],
        'color': ['E', 'E', 'E', 'I', 'J'],
        'clarity': ['SI2', 'SI1', 'VS1', 'VS2', 'SI2'],
        'depth': [61.5, 59.8, 56.9, 62.4, 63.3],
        'table': [55, 61, 65, 58, 58],
        'price': [326, 326, 327, 334, 335],
        'x': [3.95, 3.89, 4.05, 4.20, 4.34],
        'y': [3.98, 3.84, 4.07, 4.23, 4.35],
        'z': [2.43, 2.31, 2.31, 2.63, 2.75]
    }
    return pd.DataFrame(data)

def test_clean_data(diamond_data):
    model = DiamondModel(datas=diamond_data, model='LinearRegression')
    model.clean_data()
    assert not model.datas.empty
    assert 'depth' not in model.datas_dummies.columns

def test_train_model(diamond_data):
    model = DiamondModel(datas=diamond_data, model='LinearRegression')
    model.clean_data()
    mae = model.train_model()
    assert mae > 0

def test_predict(diamond_data):
    model = DiamondModel(datas=diamond_data, model='LinearRegression')
    model.clean_data()
    model.train_model()
    predictions = model.predict(model.datas_dummies.drop(columns=['price']))
    assert len(predictions) == len(model.datas_dummies)

def test_save_load_diamond_model(diamond_data, tmp_path):
    model = DiamondModel(datas=diamond_data, model='LinearRegression')
    model.clean_data()
    model.train_model()
    model_path = tmp_path / "diamond_model"
    model._save()
    
    new_model = DiamondModel(id=model.id, model='LinearRegression', folder=model_path)
    assert new_model.model is not None
    new_model.clean_data()
    new_predictions = new_model.predict(new_model.datas_dummies.drop(columns=['price']))
    assert len(new_predictions) == len(new_model.datas_dummies)
