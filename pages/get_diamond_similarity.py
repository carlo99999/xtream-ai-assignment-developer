import streamlit as st
import requests

def get_diamond_similarity():
    st.markdown("""# 📏 Get Diamond Similarity""")
    
    st.markdown("""<div style="text-align: center;">
        <h3>Find Similar Diamonds</h3>
        <p>Choose the characteristics of the diamond to find similar diamonds.</p>""",unsafe_allow_html=True)
    # Mettiamo le box per scegliere le caratteristiche del diamante per trovare i diamanti simili
    response=requests.get(f"http://localhost:8000/api/get_columns_default/",params={"id":id})
    if response.status_code!=200:
        st.error(f"An error occurred while retrieving the columns: {response.content}")
        st.stop()
    
    response=response.json()
    columns_to_use=["cut","color","clarity","carat"]
    columns=response["data"]
    col=st.columns(len(columns_to_use)+1)
    diz={k:None for k in columns.keys()}
    for i,co in zip(columns_to_use,col):
        if columns[i]!=[]:
            diz[i]=co.selectbox(i.title(),columns[i])
        else:
            diz[i]=co.number_input(i.title(),min_value=0.00,step=0.01)
            
    
    diz["n"]=col[-1].number_input("How many similar diamonds?",min_value=1,step=1)

    diz["directory"]="default_model"
    if st.button("Find Similar Diamonds"):
        url_similar_diamonds = "http://localhost:8000/api/similar_diamonds"
        response = requests.post(url_similar_diamonds, json=diz)
        if response.status_code == 200:
            similar_diamonds = response.json()["data"]
            st.write("Similar Diamonds:")
            st.dataframe(similar_diamonds, height=300,width=1300,hide_index=True)
        else:
            st.error("An error occurred while retrieving similar diamonds.")
            print(response.content)
    return True
    
    
    
    
    
    
    