from DiamondModels import DiamondModel, XGBRegressorModel, LinearRegressionModel, MLPModel,file_readers,modelling_algorithms
import pandas as pd

import os

class ModelsPipeline:
    def __init__(self, data_path: str, folder: str = "ModelsPipeline"):
        self.data_path = data_path
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.models = modelling_algorithms

    def run_pipeline(self):
        data = self._load_data(self.data_path)
        
        for model_name, model_class in self.models.items():
            print(f"Training {model_name} model...")
            model_folder = os.path.join(self.folder, model_name)
            os.makedirs(model_folder, exist_ok=True)
            
            # Initialize DiamondModel for the current model
            diamond_model = DiamondModel(data, model=model_name, folder=model_folder)
            
            # Clean data
            diamond_model.clean_data()
            
            # Train model and save it
            mae_mse = diamond_model.train_model(folder_to_save=model_folder)
            print(f"{model_name} model trained. MAE: {mae_mse['mae']}, MSE: {mae_mse['mse']}")
            
            # Save visualizations
            diamond_model.visualize_scatter_matrix(save=True, show=False,path=f'{model_folder}/Visualize')
            diamond_model.visualize_histogram(save=True, show=False,path=f'{model_folder}/Visualize')
            diamond_model.visualize_diamond_prices_by('cut', save=True, show=False,path=f'{model_folder}/Visualize')
            diamond_model.plot_predictions_vs_actual(save=True, show=False,path=f'{model_folder}/Visualize')
            print(f"{model_name} model and visualizations saved in {model_folder}")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        for extension, reader in file_readers.items():
            if data_path.endswith(extension):
                df = reader(data_path)
                for i in df.columns:
                    if i.find('Unnamed') != -1:
                        df.drop(columns=[i], inplace=True)
                return df
        raise ValueError(f"Unsupported file extension: {data_path}")

# Example usage
data_path = "data/diamonds.csv"
pipeline = ModelsPipeline(data_path)
pipeline.run_pipeline()
