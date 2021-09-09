import pickle
import pandas as pd
import numpy as np
import streamlit as st
import math
import sklearn
import plotly.express as px

######################################
# recupération des données 
######################################
@st.cache
def load_model():
    pickled_model, X_test= pickle.load(open("tuple_model_lr.pkl", 'rb'))
    return pickled_model, X_test
    


@st.cache
def load_data(filename):
    data = pd.read_csv(filename)
    return data
    
    
#############################################
# remplissage de st_selectbox
#############################################
@st.cache
def list_idClient(data):
    return data.index.values
    
    
##########################################
# informations sur le client
##########################################
@st.cache
def get_data(data, idClient):
    return data[data.index == int(idClient)]
     

@st.cache    
def getInformationsClient(data, idClient,col):
    if isinstance(data.at[int(idClient[0]), col], float):
        if math.isnan(data.at[int(idClient[0]), col]):
            return "inconnu(e)"
        else:
            return data.at[int(idClient[0]), col]
    else:
        return data.at[int(idClient[0]), col]




@st.cache
def load_prediction(data, id, clf):
        X=data
        #X = X.drop(["TARGET"], axis = 1)
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score


     
#@st.cache(allow_output_mutation=True)
def getHistogramme2(data, idClient, col,  mod, title):
    data_bis = data.copy()
    if not mod:
        val = data_bis.at[int(idClient[0]), col]

 
        fig = px.histogram(data_bis, x = col, title = title)
        
    else:
        col_a = col+"bis"
        data_bis[col_a] = np.abs(round((data_bis[col]/365), 2))
        val = data_bis.at[int(idClient[0]), col_a]

 
        fig = px.histogram(data_bis, x = col_a, title = title)
    
    fig.update_xaxes(title=dict(text=col))

    fig.add_vline(x = val, line_width=3, line_dash="dash", line_color="green")

 
    return fig
  

 

