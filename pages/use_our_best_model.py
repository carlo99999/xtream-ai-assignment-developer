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

    # Mettiamo delle box per scegliere le caratteristiche del diamante per fare le predizioni
    response=requests.get(f"http://localhost:8000/api/get_columns_default/",params={"id":id})
    if response.status_code!=200:
        st.error(f"An error occurred while retrieving the columns: {response.content}")
        st.stop()
    response=response.json()
    columns=response["data"]
    col=st.columns(len(columns))
    diz={k:None for k in columns.keys()}
    for i,co in zip(columns,col):
        if columns[i]!=[]:
            diz[i]=co.selectbox(i.title(),columns[i])
            
        else:
            diz[i]=co.number_input(i.title(),min_value=0.00,step=0.01)
    diz["directory"]="default_model"
            
    # Aggiungi un pulsante per fare le predizioni
    if st.button("Make Predictions"):
        st.write("Making predictions...")
        url = f"http://localhost:8000/api/predict/"
        response = requests.post(url, json=diz,params={"id":id})
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