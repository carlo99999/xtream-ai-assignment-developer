import streamlit as st

def use_your_trained_model():
    st.title("🚀 Use Your Trained Model")
    
    st.markdown("""
    <div style="text-align: center;">
        <h3>Select and Train Your Model</h3>
        <p>Choose a model from the dropdown menu and click the button to start training your selected model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🔧 Model Training
    """)
    
    # Aggiungi un select box per scegliere il modello
    model_options = [ "XRGBoost (Default and Suggested)","Linear Regression"]
    selected_model = st.selectbox("Choose a model to train:", model_options)
    
    # Aggiungi un pulsante per avviare il training del modello
    if st.button("Train Model"):
        st.write(f"Training the {selected_model} model...")
        
        # Qui puoi aggiungere il codice per avviare il training del modello
        # ad esempio, una funzione che si occupa del training
        # train_model(selected_model)
    
    st.markdown("""
    <hr>
    <div style="text-align: center;">
        <h4>Training Complete?</h4>
        <p>Once your model is trained, navigate to the next section to start making predictions!</p>
    </div>
    """, unsafe_allow_html=True)

def datas_not_previously_uploaded():
    st.title("🚀 Use Your Trained Model")

    st.markdown("""
    <div style="text-align: center;">
        <h3>Select and Train Your Model</h3>
        <p>First, make sure to load your dataset. You can do this in the "Load Data" section. After loading the data, choose a model from the dropdown menu and click the button to start training your selected model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🔧 Model Training
    """)

    # Aggiungi un messaggio per caricare i dati prima di addestrare il modello
    st.info("Make sure to load your dataset before training a model.")