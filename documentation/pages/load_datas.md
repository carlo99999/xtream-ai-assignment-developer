# Streamlit Function for Loading Dataset

This documentation provides an overview and explanation of the `load_datas` function for the Streamlit application designed for diamond price prediction. The function allows users to upload a dataset or use a preloaded dataset to train the diamond price prediction model.

## Overview

The `load_datas` function provides an interface for users to load a dataset for training the diamond price prediction model. Users can upload their own dataset in various formats or choose to use a preloaded dataset. The function also provides a preview of the loaded data to ensure it has been loaded correctly.

## Function: `load_datas`

This function handles the process of uploading and previewing the dataset.

```python
import streamlit as st
import requests
import io
import pandas as pd

def load_datas(id: str) -> bool:
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
    1. **Upload Your Dataset**: Click on the upload button below to upload your own dataset in CSV, EXCEL, or JSON format.
    2. **Use Preloaded Dataset**: You can choose to use our preloaded dataset if you don't have your own data.
    3. **Data Preview**: After loading the data, you can preview it below to ensure it has been loaded correctly.
    """)
    
    # Placeholder for file upload
    uploaded_file = st.file_uploader("", type=["csv", "xlsx", "json"])
    end_of_page = st.empty()
    end_of_page.markdown("""
    <hr>
    <div style="text-align: center;">
        <h4>Data Ready?</h4>
        <p>Once your data is loaded, navigate to the next section to start training the model!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save the data to the server
        url = "http://localhost:8000/api/datas"
        files = {
            "file_name": (None, uploaded_file.name),
            "id": (None, id),
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response = requests.post(url, files=files)
        if response.status_code != 200:
            print(response.content)
            st.error(f"{response.json()['message']}")
            st.stop()
        
        data = response.json()["data"]
        st.write("Data Preview:")
        st.dataframe(data, height=300, width=1300, hide_index=True)
        st.success("Data loaded successfully!")
        end_of_page.empty()
        return True
    
    return False
```

### Explanation
- **Title and Instructions**: Sets the title and provides instructions for the user.
- **How to Load Data**: Explains the steps to upload a dataset or use a preloaded dataset.
- **File Uploader**: Provides a file uploader widget to upload datasets in CSV, Excel, or JSON formats.
- **Data Preview**: After uploading, the function sends the file to the FastAPI backend, retrieves a preview of the data, and displays it in a table.
- **Success Message**: Displays a success message once the data is loaded successfully.

## Usage

This function is typically called within the main Streamlit application to load datasets for training diamond price prediction models.

### Example

```python
if st.session_state.page == "Load Datas":
    st.session_state.data_loaded = load_datas.load_datas(id=st.session_state.id)
```

### Explanation
- **Page Condition**: Checks if the current page is "Load Datas".
- **Load Data**: Calls the `load_datas` function to handle dataset loading and sets the `data_loaded` state accordingly.

## Conclusion

This documentation provides a detailed overview of the `load_datas` function. The function allows users to upload or use preloaded datasets, provides a preview of the data, and ensures a smooth setup process for training diamond price prediction models. The intuitive interface and clear instructions make it user-friendly and efficient.