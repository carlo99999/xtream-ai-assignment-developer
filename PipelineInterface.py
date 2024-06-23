import streamlit as st
from DiamondModels import DiamondModel,modelling_algorithms
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import pandas.api.types as ptypes

existing_model_ids = list(modelling_algorithms.keys())
st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "columns_to_drop" not in st.session_state:
    st.session_state.columns_to_drop=[]
if "prediction" not in st.session_state:
    st.session_state.prediction = False
if "success_string" not in st.session_state:
    st.session_state.success_string = ""


st.title("Diamond Price Prediction")



models_id={}

def center(col,n):
    for i in range(n):
        col.write("     ")

def upload_file(file_upload):
    file_extension = file_upload.name.split('.')[-1]
    if file_extension == 'csv':
        data = pd.read_csv(file_upload)
    elif file_extension in ['xlsx', 'xls']:
        data = pd.read_excel(file_upload)
    elif file_extension == 'json':
        data = pd.read_json(file_upload)
    else:
        st.error("Unsupported file type")
        st.stop()

    model = DiamondModel(datas=data, model=model_type)
    st.session_state.model = model
    st.session_state.model_trained = True
    
    st.success(f"Model created successfully with ID: {model.id} and Model: {model_type}")
    
    
def load_model(model_id,model_type):
    model = DiamondModel(id=model_id, model=model_type)
    st.session_state.model = model
    st.session_state.loaded_model = True
    st.success(f"Model {model_id} loaded successfully")
    
if not st.session_state.loaded_model and not st.session_state.model_trained:
    for i in os.listdir('models'):
        tmp=i.split('.')[0]
        tmp_split=tmp.split('_')
        id=tmp_split[0]
        type_of_model=tmp_split[1]
        if i.endswith('.pkl') and f"ID: {id} - Model: {type_of_model}" not in models_id:
            models_id[f"ID: {id} - Model: {type_of_model}"]=[id,type_of_model]  


    file_upload = st.file_uploader("Upload your dataset, remeber that you need the carat, cut, color, clarity, depth, table, price, x, y, z Columns", type=["csv", "xlsx", "json"])

    model_type = st.selectbox("Select the model type", options=["---Select---","LinearRegression", "XGBRegressor"])

    model_id = st.selectbox("Or select an existing model ID", options=["---Select---"] + list(models_id.keys()))

    if st.button("Load or Train Model"):
        if model_id != "---Select---" and file_upload is not None:
            st.error("Please upload a file OR select an existing model ID, not both")
        elif file_upload is not None and model_id=="---Select---":
            upload_file(file_upload)
        elif model_id != "---Select---" and file_upload is None:
            model_type = models_id[model_id][1]
            model_id = models_id[model_id][0]
            load_model(model_id,model_type)
        else:
            st.error("Please upload a file or select an existing model ID")

if st.session_state.model_trained and not st.session_state.loaded_model and not st.session_state.prediction:
    col1, col2 = st.columns(2)
    image_container = st.empty()
    center(col1,3)
    col1.write("Want to see the scatter Matrix?")
    with col2:
        center(col2,3)  
        if st.button("Yes",key="scatter"):
            model:DiamondModel=st.session_state.model
            fig=model.visualize_scatter_matrix()
            image_container.pyplot(plt.gcf())
            
    center(col1,3)        
    col1.write("Want to see the histogram matrix?")
    with col2:
        center(col2,2)
        if st.button("Yes",key="histogram"):
            model:DiamondModel=st.session_state.model
            fig=model.visualize_histogram()
            image_container.pyplot(plt.gcf())
    
    center(col1,3)
    col1.write("Want to see the diamond prices by a column?")
    feature=col1.selectbox("Select the column",options=list(st.session_state.model.datas.columns))
    with col2:
        center(col2,2)
        if st.button("Yes",key="violin"):
            model:DiamondModel=st.session_state.model
            fig=model.visualize_diamond_prices_by(feature)
            image_container.pyplot(plt.gcf(),)
            
    center(col1,1)
    
    list_of_columns=[col for col in ["Use default setting","Don't want to drop any column"]+list(st.session_state.model.datas.columns) if col not in st.session_state.columns_to_drop]
    
    st.markdown("If you want to drop a column, please click the button below")  
    st.markdown("If you don't want to drop any column, please select 'Don't want to drop any column'")
    
    column_to_drop=st.selectbox("Select the column to drop",options=list_of_columns)
    
    if st.button("Drop this column",key="train"):
        if column_to_drop=="Don't want to drop any column":
            st.session_state.columns_to_drop=[]
        elif column_to_drop=="Use default setting" or column_to_drop=="Use default setting":
            st.session_state.columns_to_drop=["Use default setting"]
        elif column_to_drop!="---Select---":
            st.session_state.columns_to_drop.append(column_to_drop)
        
    if st.session_state.columns_to_drop!=[] and st.session_state.columns_to_drop!=["Default setting"]:
        st.write(f"Columns to drop: {"   ".join(st.session_state.columns_to_drop)}")
        
    if st.button("Train the model",key="train_model"):
        if column_to_drop=="Use default setting":
            st.session_state.columns_to_drop=["Use default setting"]
        if st.session_state.columns_to_drop==[]:
            st.session_state.model.clean_data(columns_to_drop=[])
            mae=st.session_state.model.train_model()
        elif st.session_state.columns_to_drop==["Use default setting"]:
            st.session_state.model.clean_data()
            mae=st.session_state.model.train_model()
        else:
            st.session_state.model.clean_data(columns_to_drop=st.session_state.columns_to_drop)
            mae=st.session_state.model.train_model()
        st.session_state.success_string=f"Model trained successfully with MAE: {mae:.2f}"
        st.session_state.prediction=True

if st.session_state.loaded_model and not st.session_state.model_trained:
    st.session_state.success_string=f"Model loaded successfully"
    st.session_state.prediction=True
    

if st.session_state.prediction and st.session_state.model!=None:
    st.markdown(f"## {st.session_state.success_string}")
    st.markdown("### Now is time to make a prediction")
    model:DiamondModel=st.session_state.model
    if st.session_state.columns_to_drop==["Use default setting"]:
        st.session_state.columns_to_drop=['depth', 'table', 'y', 'z']
        
    columns_not_dropped=list(set(model.datas_processed.columns)-set(st.session_state.columns_to_drop))
    columns_not_dropped.remove('price')
    added={k:list(set(model.datas[k].to_list())) for k in columns_not_dropped if ptypes.is_string_dtype(model.datas[k])}
    get={
        
    }
    st.write("Please insert the following values")
    
    col=st.columns(len(columns_not_dropped))
    for i in columns_not_dropped:
        with col[columns_not_dropped.index(i)]:
            if ptypes.is_string_dtype(model.datas[i]):
                get[i]=st.selectbox(i.title(),options=added[i])
            else:
               get[i]= st.number_input(i.title(),min_value=0.00,step=0.01)
    if st.button("Predict",key="predict_value"):
        df=pd.DataFrame([],columns=model.datas_dummies.columns)
        for i in get:
            if i in i in ['x','carat']:
                if get[i]<=0.0:
                    st.error(f"Please insert a valid value for {i}")
                    st.stop()
        df_tmp=pd.DataFrame(get,index=[0])
        df_tmp=pd.get_dummies(df_tmp,columns=["cut","color","clarity"])
        for i in df_tmp.columns:
        
            if i in df.columns:
                df[i]=df_tmp[i]
        df.fillna(False,inplace=True)
        df.drop(columns=["price"],inplace=True)
        prediction=model.predict(df)
        st.markdown(f"""
        ✨ **The probable market value of the diamond is {prediction[0]:.2f}** ✨

        💎 This value is calculated based on extensive analysis and our advanced machine learning model. 💎

        _Thank you for using our Diamond Price Prediction service!_
        """)

          
        
                
    
    
    
if st.button("Go back",key="save"):
    for i in st.session_state:
        if i in ["loaded_model","model_trained","model","prediction","success_string"]:
            st.session_state[i]=None
        if i=="columns_to_drop":
            st.session_state.columns_to_drop=[]
    st.rerun()