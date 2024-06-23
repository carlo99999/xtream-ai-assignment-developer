# Streamlit Function for Predicting with a Pretrained Model

This documentation provides an overview and explanation of the `predict_with_our_model` function for the Streamlit application designed for diamond price prediction. The function allows users to make predictions using a pretrained model by specifying diamond characteristics.

## Overview

The `predict_with_our_model` function provides an interface for users to input diamond characteristics and use a pretrained model to predict the diamond's price. The function retrieves the required input fields from the backend, allows users to fill in the details, and sends the data to the backend for prediction.

## Function: `predict_with_our_model`

This function handles the process of predicting diamond prices using a pretrained model.

```python
import streamlit as st
import requests

def predict_with_our_model(id):
    st.title("🔮 Predict with Our Pretrained Model")

    st.markdown("""
    <div style="text-align: center;">
        <h3>Make Predictions with Our Best Model</h3>
        <p>Choose a model from the dropdown menu and upload a dataset to make predictions using the trained model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🔮 Make Predictions
    """)

    # Retrieve columns for default diamond attributes
    response = requests.get(f"http://localhost:8000/api/get_columns_default/", params={"id": id})
    if response.status_code != 200:
        st.error(f"An error occurred while retrieving the columns: {response.content}")
        st.stop()
    
    response = response.json()
    columns = response["data"]
    col = st.columns(len(columns))
    diz = {k: None for k in columns.keys()}
    
    # Input fields for diamond characteristics
    for i, co in zip(columns, col):
        if columns[i] != []:
            diz[i] = co.selectbox(i.title(), columns[i])
        else:
            diz[i] = co.number_input(i.title(), min_value=0.00, step=0.01)
    
    diz["directory"] = "default_model"
    
    # Button to make predictions
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
- **Title and Instructions**: Sets the title and provides instructions for the user.
- **Retrieve Columns Request**: Sends a GET request to retrieve the columns of the default diamond attributes.
- **Error Handling**: Checks the response status and displays an error message if the request fails.
- **Input Fields**: Creates input fields for the user to specify diamond characteristics (e.g., cut, color, clarity, carat).
- **Prediction Button**: Sends a POST request with the user input to retrieve the predicted diamond price and displays the result.

## Usage

This function is typically called within the main Streamlit application to predict diamond prices using pretrained models.

### Example

```python
if st.session_state.page == "Use Our Best Models":
    use_our_best_model.predict_with_our_model("Default")
```

### Explanation
- **Page Condition**: Checks if the current page is "Use Our Best Models".
- **Predict with Our Model**: Calls the `predict_with_our_model` function to handle predictions using the pretrained model.

## Conclusion

This documentation provides a detailed overview of the `predict_with_our_model` function. The function allows users to input diamond characteristics and make predictions using a pretrained model, providing an intuitive and user-friendly interface for interacting with the diamond price prediction model. The clear instructions and error handling ensure a smooth and efficient user experience.