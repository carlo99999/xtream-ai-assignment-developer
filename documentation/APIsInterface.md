# Streamlit Application for Diamond Price Prediction with Navigation

This documentation provides an overview and explanation of a Streamlit application designed for diamond price prediction. The application includes various functionalities such as loading data, using trained models, and finding similar diamonds, with a navigation bar to switch between different pages.

## Overview

The application includes the following features:
- Load and train diamond price prediction models.
- Use trained models to make predictions.
- Find similar diamonds based on specified attributes.
- Navigate through different pages using a navigation bar.

## Imports and Setup

The necessary libraries and modules are imported at the beginning of the script.

```python
import requests
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from streamlit_navigation_bar import st_navbar
from options.navbar_option import options, styles, logo, pages
from pages import home, load_datas, use_your_trained_model, use_our_best_model, about, get_diamond_similarity
import uuid
```

### Explanation of Imports
- **Requests (`requests`)**: For making HTTP requests to the backend.
- **Streamlit (`st`)**: For creating the web application interface.
- **Pandas (`pd`) and Numpy (`np`)**: For data manipulation and analysis.
- **OS (`os`) and JSON (`json`)**: For interacting with the operating system and handling JSON data.
- **Streamlit Navigation Bar (`st_navbar`)**: For adding a navigation bar to the Streamlit app.
- **Custom Modules (`pages` and `options`)**: Custom modules containing different pages and navigation bar options.

## Session State Initialization

The application initializes various session state variables to manage its state.

```python
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4().hex

st.set_page_config(layout="wide", page_title="Diamond Price Prediction", page_icon="💎", initial_sidebar_state="collapsed")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "trained_model" not in st.session_state:
    st.session_state.trained_model = True
```

### Explanation
- **Unique ID**: Generates a unique ID for the session.
- **Page Configuration**: Sets the page layout, title, and icon.
- **Session State Variables**: Initializes variables to track data loading, current page, and trained model status.

## Navigation Bar

The navigation bar is set up to allow users to switch between different pages.

```python
st.session_state.page = st_navbar(pages=pages, options=options, styles=styles, logo_path=logo)
```

### Explanation
- **Navigation Bar**: Uses the `st_navbar` function to create a navigation bar with the specified pages, options, styles, and logo.

## Page Routing

The application routes to different pages based on the selected navigation option.

### Home Page

```python
if st.session_state.page == "Home":
    home.home()
```

### Load Datas Page

```python
if st.session_state.page == "Load Datas":
    st.session_state.data_loaded = load_datas.load_datas(id=st.session_state.id)
```

### Use Your Trained Model Page

```python
if st.session_state.page == "Use Your Trained Model":
    if st.session_state.data_loaded:
        st.session_state.trained_model = use_your_trained_model.use_your_trained_model(st.session_state.id)
    if st.session_state.trained_model:
        st.session_state.trained_model = use_your_trained_model.predict_with_trained_model("ddb95b618c1349308f31b07154e3c1da")
    else:
        use_your_trained_model.datas_not_previously_uploaded()
```

### Use Our Best Models Page

```python
if st.session_state.page == "Use Our Best Models":
    use_our_best_model.predict_with_our_model("Default")
```

### Get Diamond Similarity Page

```python
if st.session_state.page == "Get Diamond Similarity":
    get_diamond_similarity.get_diamond_similarity()
```

### About Page

```python
if st.session_state.page == "About":
    about.about()
```

### Explanation
- **Home Page**: Displays the home page.
- **Load Datas Page**: Allows users to load data for training models.
- **Use Your Trained Model Page**: Allows users to train and use their own models for predictions.
- **Use Our Best Models Page**: Allows users to use pre-trained models for predictions.
- **Get Diamond Similarity Page**: Allows users to find diamonds similar to a specified diamond.
- **About Page**: Displays information about the application.

## Conclusion

This documentation provides a detailed overview of the Streamlit application for diamond price prediction. The application is structured to provide an intuitive interface for users to load data, train models, make predictions, and find similar diamonds. The navigation bar enables easy switching between different functionalities, making the application user-friendly and efficient.
