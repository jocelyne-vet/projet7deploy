#import score
import app

import streamlit as st 
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import sklearn
import plotly.express as px
import plotly.graph_objects as go
import imblearn
import math

#################################################

clf, X_test = app.load_model()
data = X_test
#data = app.load_data("X_sample.csv") 
################################################
def show_score(score, seuil):
    
    st.markdown("**Prédiction : **") 
    if score < seuil:
        st.text("Solvable")
    else:
        st.text("Non solvable")
    
    
    
def show_probabilite(score, seuil):
    st.markdown("**Score du client: **") 
    # st.text(score)
    
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = score,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score", 'font': {'size': 24}},
    # delta = {'reference': 400, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, seuil], 'color': 'cyan'},
            {'range': [seuil, 1], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': seuil}}))

    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"}, width = 600, height = 400)



    st.plotly_chart(fig, width = 200)






##############################################

sel_col, display_col = st.columns(2)

add_selectbox = st.sidebar.container()

with add_selectbox:
    idClient = st.selectbox("Id du client: ", 
                         app.list_idClient(data)) 
    #st.text("Il y a 3 Clients dans la base")

    visualisation = st.radio("Informations: ", ('Informations relatives au score', 'Informations relatives au client',
                             'Informations relatives aux features')) 

resultats = st.container()





with resultats :
    if (visualisation == 'Informations relatives au score'):

        st.title("Informations relatives au score")
      
        # print('idClient', idClient)
        # if not math.isnan(idClient):
        score_res = app.load_prediction(data, idClient, clf)     
            
        result_score = st.container()                   
        with result_score:
            my_range = np.arange(0, 1, 0.01)
            seuil = 0.5
            number = st.select_slider("Seuil :", options = my_range, value = seuil)
            col1, col2 = st.columns(2)
            cont1 = col1.container()
            with cont1:
                show_probabilite(score_res[0], number)
            cont2 = col1.container()
            with cont2:
                show_score(score_res[0],number)
        
            
            
           
            
    #else:       
    elif (visualisation == 'Informations relatives au client'):
        # if  len(idClient):
        st.title("Informations relatives au client")
            
        st.write("**Sexe : **", app.getInformationsClient(data, idClient,"CODE_GENDER"))
        st.write("**Age : **{:.0f} ans".format(np.abs(int(app.getInformationsClient(data, idClient,"DAYS_BIRTH")/365))))
        st.write("**Statut familial : **", app.getInformationsClient(data, idClient,"NAME_FAMILY_STATUS"))
        st.write("**Nombre d'enfants : **{:.0f}".format(app.getInformationsClient(data, idClient,"CNT_CHILDREN")))
        st.write("**La catégorie socio professionnelle : **", app.getInformationsClient(data, idClient,"NAME_INCOME_TYPE"))
        st.write("**La profession : **", app.getInformationsClient(data, idClient,"OCCUPATION_TYPE"))
        st.write("**Le revenu : **", app.getInformationsClient(data, idClient,"AMT_INCOME_TOTAL"))
        
        st.markdown('*Dans les graphiques ci-dessous la ligne verte, en pointillés, représente le client.*') 
        st.plotly_chart(app.getHistogramme2(data, idClient, "DAYS_BIRTH", True, "Distribution: âge des clients"))
        st.plotly_chart(app.getHistogramme2(data, idClient, "DAYS_EMPLOYED",  True, "Distribution: ancienneté des clients"))
        st.plotly_chart(app.getHistogramme2(data, idClient, "AMT_INCOME_TOTAL",  False, "Distribution: revenu des clients"))
        st.plotly_chart(app.getHistogramme2(data, idClient, "AMT_CREDIT",  False, "Distribution: annuité du crédit"))
   
    else:
        result_features = st.container()
        with result_features:
        
            st.title("Features importance")
                
            # images = ['features_imp_shap1.png', 'features_imp_shap2.png']
            # st.image(images, use_column_width=True, caption=["some generic text"] * len(images))
             
            st.write("Voici les 10 coefficients ayant le plus large positif impact sur la probabilité d'être non solvable.")
            
            st.image('figure_3.png', width=700)
                
            st.write("Voici les 10 coefficients ayant le plus large négatif impact sur la probabilité d'être non solvable.")
            
            st.image('figure_4.png', width=700)
                
            st.write("Voici l'importance des features.")
            
            
            
            st.write("Pour résumer, la caractéristique la plus forte dans l’ensemble de données de ce jeu est le ratio new_credit_to_goods_ratio. Une augmentation de ce ratio d’une unité augmente les chances d’être de la classe non solvable d’un facteur de 100 lorsque toutes les autres caractéristiques restent les mêmes.")

            st.image('features_imp.png', width=700)
    