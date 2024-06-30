from typing import List

DATA_FOLDER = "data"
MODELS_FOLDER = "models"
VISUALIZATIONS_FOLDER = "visualizations"

COLUMNS_TO_DROP: List[str] = ['depth', 'table', 'y', 'z']
COLUMNS_TO_DUMMIES: List[str] = ['cut', 'color', 'clarity']