# Streamlit Application for Using and Predicting with Trained Diamond Models

This documentation provides an overview and explanation of a Streamlit application designed to train and predict diamond prices using trained models. The application interacts with a FastAPI backend to handle training and prediction tasks.

## Overview

The application includes functionalities to:
- Train a diamond price prediction model.
- Make predictions using a trained model.

## Imports and Setup

The necessary libraries and modules are imported at the beginning of the script.

```python
import streamlit as st
import requests
from DiamondModels import DiamondModel, modelling_algorithms
import os
```

### Explanation of Imports
- **Streamlit (`st`)**: For creating the web application interface.
- **Requests (`requests`)**: For making HTTP requests to the FastAPI backend.
- **DiamondModels**: Custom module containing diamond model classes and algorithms.
- **OS (`os`)**: For interacting with the operating system.

## Function: `use_your_trained_model`

This function handles the training of a model using the selected algorithm.

```python
def use_your_trained_model(id: str):
    if "trained_button" not in st.session_state:
        st.session_state.trained_button = False
    st.title("🚀 Use Your Trained Model")
    
    st.markdown("""
    <div style="text-align: center;">
        <h3>Select and Train Your Model</h3>
        <p>Choose a model from the dropdown menu and click the button to start training your selected model.</p>
        <p>We suggest you use the XGBRegressor model for better accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔧 Model Training")
    
    model_options = modelling_algorithms.keys()
    selected_model = st.selectbox("Choose a model to train:", model_options)
    
    if st.button("Train Model") and not st.session_state.trained_button:
        st.session_state.trained_button = True
    if st.session_state.trained_button:
        st.write(f"Training the {selected_model} model...")
        url = "http://localhost:8000/api/train/"
        data = {"model_type": selected_model, "id": id}
        with st.spinner("Training the model..."):
            response = requests.post(url, data=data)
        if response.status_code == 200:
            st.success("Model trained successfully!")
            return True
        else:
            st.error("An error occurred while training the model.")
            st.session_state.trained_button = False
            return False
        st.session_state.trained_button = False
    return False
```

### Explanation
- **Title and Instructions**: Sets the title and instructions for the user.
- **Model Selection**: Provides a dropdown menu for selecting a model.
- **Train Button**: Initiates the training process when clicked.
- **Training Request**: Sends a POST request to the FastAPI backend to train the selected model.

## Function: `datas_not_previously_uploaded`

This function displays a message if data has not been uploaded before attempting to train a model.

```python
def datas_not_previously_uploaded():
    st.title("🚀 Use Your Trained Model")

    st.markdown("""
    <div style="text-align: center;">
        <h3>Select and Train Your Model</h3>
        <p>First, make sure to load your dataset. You can do this in the "Load Data" section. After loading the data, choose a model from the dropdown menu and click the button to start training your selected model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🔧 Model Training")

    st.info("Make sure to load your dataset before training a model.")
    return False
```

### Explanation
- **Title and Instructions**: Sets the title and instructions for the user.
- **Info Message**: Informs the user to upload data before training a model.

## Function: `predict_with_trained_model`

This function handles making predictions with a trained model.

```python
def predict_with_trained_model(id):
    st.title("🔮 Predict with Trained Model")

    st.markdown("""
    <div style="text-align: center;">
        <h3>Make Predictions with Your Trained Model</h3>
        <p>Choose a model from the dropdown menu and upload a dataset to make predictions using the trained model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🔮 Make Predictions")

    response = requests.get(f"http://localhost:8000/api/get_columns_trained/", params={"id": id})
    if response.status_code != 200:
        st.error(f"An error occurred while retrieving the columns: {response.content}")
        st.stop()
    response = response.json()
    columns = response["data"]
    col = st.columns(len(columns))
    diz = {k: None for k in columns.keys()}
    for i, co in zip(columns, col):
        if columns[i] != []:
            diz[i] = co.selectbox(i.title(), columns[i])
        else:
            diz[i] = co.number_input(i.title(), min_value=0.00, step=0.01)
    
    diz['directory'] = 'trained_models'
    
    if st.button("Make Predictions"):
        st.write("Making predictions...")
        url = f"http://localhost:8000/api/predict/"
        response = requests.post(url, json=diz, params={"id": id})
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.markdown(f"""
        ✨ **The probable market value of the diamond is {prediction[0]:.2f}** ✨

        💎 This value is calculated based on extensive analysis and our advanced machine learning model. 💎

        _Thank you for using our Diamond Price Prediction service!_
        """)
        else:
            st.error("An error occurred while making predictions.")
            print(response.content)

    return True
```

### Explanation
- **Title and Instructions**: Sets the title and instructions for the user.
- **Retrieve Columns**: Sends a GET request to retrieve the columns of the uploaded data.
- **Input Fields**: Creates input fields for the user to enter diamond attributes.
- **Prediction Request**: Sends a POST request to the FastAPI backend to make predictions with the trained model.

## Conclusion

This documentation provides an overview of the Streamlit application functions for training and predicting diamond prices using trained models. The application interacts with a FastAPI backend to handle model training and prediction tasks, providing an intuitive interface for users to utilize machine learning models effectively.