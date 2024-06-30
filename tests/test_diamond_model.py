import pytest
import pandas as pd
import numpy as np
from DiamondModels import DiamondModel, LinearRegressionModel, XGBRegressorModel, MLPModel

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'carat': [0.23, 0.21, 0.23, 0.29, 0.31],
        'cut': ['Ideal', 'Premium', 'Good', 'Premium', 'Good'],
        'color': ['E', 'E', 'E', 'I', 'J'],
        'clarity': ['SI2', 'SI1', 'VS1', 'VS2', 'SI2'],
        'depth': [61.5, 59.8, 62.0, 62.4, 63.3],
        'table': [55.0, 61.0, 58.0, 58.0, 58.0],
        'price': [326, 326, 327, 334, 335],
        'x': [3.95, 3.89, 4.05, 4.20, 4.34],
        'y': [3.98, 3.84, 4.07, 4.23, 4.35],
        'z': [2.43, 2.31, 2.31, 2.63, 2.75]
    })

@pytest.fixture
def diamond_model(sample_data):
    return DiamondModel(datas=sample_data, model='LinearRegression')

def test_init(sample_data):
    model = DiamondModel(datas=sample_data, model='LinearRegression')
    assert isinstance(model, DiamondModel)
    assert isinstance(model.model, LinearRegressionModel)

def test_clean_data(diamond_model):
    diamond_model.clean_data()
    assert 'depth' not in diamond_model.datas_processed.columns
    assert 'cut_Premium' in diamond_model.datas_dummies.columns
    assert diamond_model.datas_dummies.shape[1] > diamond_model.datas_processed.shape[1]

def test_train_model(diamond_model):
    diamond_model.clean_data()
    mae_mse = diamond_model.train_model()
    assert 'mae' in mae_mse
    assert 'mse' in mae_mse
    assert mae_mse['mae'] > 0
    assert mae_mse['mse'] > 0

def test_predict(diamond_model):
    diamond_model.clean_data()
    diamond_model.train_model()
    new_data = pd.DataFrame({
        'carat': [0.25],
        'cut': ['Ideal'],
        'color': ['E'],
        'clarity': ['VS2'],
        'depth': [62.2],
        'table': [57.0],
        'x': [4.01],
        'y': [4.06],
        'z': [2.50]
    })
    prediction = diamond_model.predict(new_data)
    assert isinstance(prediction, pd.Series)
    assert len(prediction) == 1
    assert prediction.iloc[0] > 0

def test_get_similar_samples(diamond_model):
    similar_samples = diamond_model.get_similar_samples(carat=0.23, cut='Ideal', color='E', clarity='SI2', n=2)
    assert isinstance(similar_samples, pd.DataFrame)
    assert len(similar_samples) == 2
    assert all(similar_samples['cut'] == 'Ideal')
    assert all(similar_samples['color'] == 'E')
    assert all(similar_samples['clarity'] == 'SI2')

def test_prepare_data_for_prediction(sample_data):
    model = DiamondModel(datas=sample_data, model='LinearRegression')
    model.clean_data()
    model.train_model()
    
    new_data = pd.DataFrame({
        'carat': [0.25],
        'cut': ['Ideal'],
        'color': ['E'],
        'clarity': ['VS2']
    })
    
    prepared_data = DiamondModel.prepare_data_for_prediction(model.datas_dummies, new_data)
    assert 'cut_Ideal' in prepared_data.columns
    assert 'color_E' in prepared_data.columns
    assert 'clarity_VS2' in prepared_data.columns
    assert prepared_data.shape[1] == model.datas_dummies.drop('price', axis=1).shape[1]

def test_plot_gof(diamond_model):
    diamond_model.clean_data()
    diamond_model.train_model()
    y_true = diamond_model.GT_Y
    y_pred = diamond_model.predictions
    fig = DiamondModel.plot_gof(y_true, y_pred, show=False, save=False)
    assert fig is not None

@pytest.mark.parametrize("model_name", ["LinearRegression", "XGBRegressor", "MLP"])
def test_different_models(sample_data, model_name):
    model = DiamondModel(datas=sample_data, model=model_name)
    model.clean_data()
    mae_mse = model.train_model()
    assert 'mae' in mae_mse
    assert 'mse' in mae_mse

def test_save_and_load(diamond_model, tmp_path):
    diamond_model.clean_data()
    diamond_model.train_model()
    
    save_path = tmp_path / "test_model"
    diamond_model._save(str(save_path))
    
    loaded_model = DiamondModel(id=diamond_model.id, folder=str(save_path))
    assert loaded_model.model_name == diamond_model.model_name
    assert loaded_model.datas.equals(diamond_model.datas)
    assert loaded_model.datas_dummies.equals(diamond_model.datas_dummies)

def test_visualize_scatter_matrix(diamond_model):
    fig = diamond_model.visualize_scatter_matrix(save=False, show=False)
    assert fig is not None

def test_visualize_histogram(diamond_model):
    fig = diamond_model.visualize_histogram(save=False, show=False)
    assert fig is not None

def test_visualize_diamond_prices_by(diamond_model):
    fig = diamond_model.visualize_diamond_prices_by('cut', save=False, show=False)
    assert fig is not None

def test_error_handling():
    with pytest.raises(ValueError):
        DiamondModel()
    
    with pytest.raises(ValueError):
        DiamondModel(datas=pd.DataFrame())

    with pytest.raises(ValueError):
        DiamondModel(datas=pd.DataFrame(), model='InvalidModel')

def test_get_similar_samples_error_handling(diamond_model):
    with pytest.raises(ValueError):
        diamond_model.get_similar_samples(carat=-1, cut='Ideal', color='E', clarity='SI2', n=2)
    
    with pytest.raises(ValueError):
        diamond_model.get_similar_samples(carat=0.23, cut='InvalidCut', color='E', clarity='SI2', n=2)