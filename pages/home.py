import streamlit as st

def home():
    st.title("💎 Diamond Price Prediction")
    
    st.markdown("""
    <div style="text-align: center;">
        <h3>Welcome to the Diamond Price Prediction App</h3>
        <p>Use this app to predict the price of a diamond based on its features.</p>
        <p>Please use the navigation bar to the left to navigate to the different pages of the app.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🌟 How does it work?
    """)
    
    st.markdown("""
    1. **Load Datas**: You can decide to load the data you want to use to train the model, or you can use our pretrained models.
    2. **Use Our Best Models**: You can use our (pre)trained models to predict the price of a diamond.
    3. **Get Diamond Similarity**: You can use this feature to get the similarity of a diamond to the ones in the dataset.
    """)
    
    st.markdown("""
    <hr>
    <div style="text-align: center;">
        <h4>Ready to get started?</h4>
        <p>Navigate to the different sections using the sidebar to start predicting diamond prices!</p>
    </div>
    """, unsafe_allow_html=True)