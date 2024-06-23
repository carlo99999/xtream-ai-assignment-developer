import requests
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from streamlit_navigation_bar import st_navbar
from options.navbar_option import options, styles,logo,pages
from pages import home,load_datas, use_your_trained_model, use_our_best_model, about,get_diamond_similarity
import uuid

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4().hex

st.set_page_config(layout="wide", page_title="Diamond Price Prediction", page_icon="💎",initial_sidebar_state="collapsed")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "trained_model" not in st.session_state:
    st.session_state.trained_model = True

st.session_state.page = st_navbar(pages=pages, options=options, styles=styles, logo_path=logo)
if st.session_state.page == "Home":
    home.home()
    
if st.session_state.page=="Load Datas":
    st.session_state.data_loaded = load_datas.load_datas(id=st.session_state.id)
    
    
if st.session_state.page=="Use Your Trained Model":
    if st.session_state.data_loaded:
        st.session_state.trained_model = use_your_trained_model.use_your_trained_model(st.session_state.id)
    if st.session_state.trained_model:
        st.session_state.trained_model = use_your_trained_model.predict_with_trained_model("ddb95b618c1349308f31b07154e3c1da")
    else:
        use_your_trained_model.datas_not_previously_uploaded()
        
if st.session_state.page=="Use Our Best Models":
    use_our_best_model.predict_with_our_model("Default")
    
if st.session_state.page=="Get Diamond Similarity":
    get_diamond_similarity.get_diamond_similarity()
    
if st.session_state.page=="About":
    about.about()


    
