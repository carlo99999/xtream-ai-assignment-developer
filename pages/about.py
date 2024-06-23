import streamlit as st
import requests

def about():
    st.title("📚 About 💎 Diamond Price Prediction App")
    
    st.markdown("""
    <div style="text-align: center;">
        <h3>About Our Diamond Price Prediction App</h3>
        <p>Welcome to the Diamond Price Prediction App, your comprehensive tool for estimating and analyzing diamond prices based on their unique characteristics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📈 What We Offer
    Our app provides several functionalities to help you understand and predict diamond prices:
    
    ### 1. Load Datas
    Easily load your own dataset in CSV, Excel, or JSON format. You can also choose to use our preloaded dataset for a quicker setup.
    
    ### 2. Use Our Best Models
    Leverage our pretrained machine learning models to predict the price of a diamond based on its features. Simply input the diamond's characteristics and receive an estimated market value.

    ### 3. Get Diamond Similarity
    Find diamonds similar to your specified characteristics to understand their value and market trends. This feature helps you compare and analyze diamonds based on key attributes like cut, color, clarity, and carat.

    ### 4. Use Your Trained Model
    Train your own model using your dataset. Select from various machine learning algorithms to tailor the prediction model to your specific needs. Our app guides you through the process of training and deploying your custom model.

    ### 5. Predict with Trained Model
    Once you've trained your model, you can use it to make predictions on new data. This feature is especially useful for applying your custom-trained model to different datasets and scenarios.

    ## 🔍 How It Works
    ### Load Datas
    - **Upload Your Dataset**: Click on the upload button to upload your dataset in CSV, Excel, or JSON format.
    - **Use Preloaded Dataset**: Choose our preloaded dataset if you don't have your own data.
    - **Data Preview**: Preview the loaded data to ensure it is correctly uploaded.

    ### Predict with Our Model
    - **Select Features**: Choose the diamond's features such as cut, color, clarity, and carat.
    - **Make Predictions**: Click on the button to get an estimated market value using our advanced machine learning model.

    ### Get Diamond Similarity
    - **Specify Characteristics**: Choose the attributes of the diamond.
    - **Find Similar Diamonds**: Get a list of diamonds that are similar to the specified characteristics.

    ### Use Your Trained Model
    - **Select Model**: Choose a machine learning algorithm from the dropdown menu.
    - **Train Model**: Click the button to start training your selected model. We recommend using the XGBRegressor model for better accuracy.

    ### Predict with Trained Model
    - **Input Features**: Enter the diamond's features.
    - **Make Predictions**: Use your trained model to get the estimated price.

    ## 🌟 Why Use Our App?
    Our Diamond Price Prediction App is designed to provide accurate and reliable price estimates based on extensive data analysis and advanced machine learning models. Whether you are a jeweler, a buyer, or someone interested in diamond prices, our app can help you make informed decisions.

    Thank you for choosing our service. We hope it helps you in your diamond pricing endeavors!
    """)

# Include this function call in your main app script to add the "About" page to the navigation
if __name__ == "__main__":
    about()
