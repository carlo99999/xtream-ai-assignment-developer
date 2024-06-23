import streamlit as st
import requests
import io
import pandas as pd


def load_datas(id:str)->bool:
    st.title("📊 Load Datas")
    
    st.markdown("""
    <div style="text-align: center;">
        <h3>Load Your Dataset</h3>
        <p>Use this page to load the dataset you want to use to train the diamond price prediction model.</p>
        <p>You can also choose to use our preloaded dataset for quicker setup.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📥 How to Load Data
    """)
    
    st.markdown("""
    1. **Upload Your Dataset**: Click on the upload button below to upload your own dataset in CSV, EXCEL or JSON format.
    2. **Use Preloaded Dataset**: You can choose to use our preloaded dataset if you don't have your own data.
    3. **Data Preview**: After loading the data, you can preview it below to ensure it has been loaded correctly.
    """)
    
    # Placeholder for file upload
    uploaded_file = st.file_uploader("", type=["csv","xlsx","json",],)
    end_of_page=st.empty()
    end_of_page.markdown("""
    <hr>
    <div style="text-align: center;">
        <h4>Data Ready?</h4>
        <p>Once your data is loaded, navigate to the next section to start training the model!</p>
    </div>
    """, unsafe_allow_html=True)
    if uploaded_file is not None:
        ### Save the data to the server
        url = "http://localhost:8000/api/datas"
        files = {
            "file_name": (None, uploaded_file.name),
            "id": (None, id),
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response=requests.post(url, files=files)
        if response.status_code!=200:
            print(response.content)
            st.error(f"{response.content['message']}")
            st.stop()
        data=response.json()["data"]
        st.write("Data Preview:")
        st.dataframe(data, height=300,width=1300,hide_index=True)
        st.success("Data loaded successfully!")
        end_of_page.empty()
        return True
    return False
        