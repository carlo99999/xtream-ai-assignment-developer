import pytest
import pandas as pd
from DiamondModels import DiamondModel, LinearRegressionModel, XGBRegressorModel  # Assicurati che il percorso sia corretto
import os

# Fixture per il setup dei dati di test
@pytest.fixture
def sample_data():
    data = {
        'carat': [0.23, 0.21, 0.23, 0.29, 0.31],
        'cut': ['Ideal', 'Premium', 'Good', 'Premium', 'Good'],
        'color': ['E', 'E', 'E', 'I', 'J'],
        'clarity': ['SI2', 'SI1', 'VS1', 'VS2', 'SI2'],
        'depth': [61.5, 59.8, 56.9, 62.4, 63.3],
        'table': [55, 61, 65, 58, 58],
        'price': [326, 326, 327, 334, 335],
        'x': [3.95, 3.89, 4.05, 4.2, 4.34],
        'y': [3.98, 3.84, 4.07, 4.23, 4.35],
        'z': [2.43, 2.31, 2.31, 2.63, 2.75]
    }
    return pd.DataFrame(data)

# Parametrizzazione dei test per includere entrambi i modelli
@pytest.mark.parametrize("model_name", ["LinearRegression", "XGBRegressor"])
def test_clean_data(sample_data, model_name):
    model = DiamondModel(datas=sample_data, model=model_name)
    model.clean_data()
    assert 'depth' not in model.datas_dummies.columns
    assert 'table' not in model.datas_dummies.columns
    assert 'y' not in model.datas_dummies.columns
    assert 'z' not in model.datas_dummies.columns

@pytest.mark.parametrize("model_name", ["LinearRegression", "XGBRegressor"])
def test_train_model(sample_data, model_name):
    model = DiamondModel(datas=sample_data, model=model_name)
    mae_mse = model.train_model()
    assert 'mae' in mae_mse
    assert 'mse' in mae_mse

@pytest.mark.parametrize("model_name", ["LinearRegression", "XGBRegressor"])
def test_predict(sample_data, model_name):
    model = DiamondModel(datas=sample_data, model=model_name)
    model.train_model()
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)

@pytest.mark.parametrize("model_name", ["LinearRegression", "XGBRegressor"])
def test_save_load_diamond_model(sample_data, model_name, tmp_path):
    model = DiamondModel(datas=sample_data, model=model_name)
    model.train_model()
    model._save(folder_to_save=tmp_path)

    loaded_model = DiamondModel(id=model.id, folder=tmp_path)
    loaded_model._load(model.id)

    if model_name == "LinearRegression":
        assert isinstance(loaded_model.model, LinearRegressionModel)
    else:
        assert isinstance(loaded_model.model, XGBRegressorModel)

    assert loaded_model.datas.equals(model.datas)

@pytest.mark.parametrize("model_name", ["LinearRegression", "XGBRegressor"])
def test_get_similar_samples(sample_data, model_name):
    model = DiamondModel(datas=sample_data, model=model_name)
    similar_samples = model.get_similar_samples(carat=0.23, cut='Ideal', color='E', clarity='SI2', n=2)
    assert len(similar_samples) <= 2
    assert 'carat' in similar_samples.columns

if __name__ == "__main__":
    pytest.main()
