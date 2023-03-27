import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_resource
def load_database():
    return pd.read_excel('BRCidadesRegiao.xlsx').query('estado in ["DF", "BA", "ES"]')

def highlight_class_lc(s):
    if s.outlier_max == 1:
        return ['background-color: #00cec9']*len(s)
    elif s.class_lc == 'acima':
        return ['background-color: #fab1a0']*len(s)
    elif s.class_lc == 'media':
        return ['background-color: #ffeaa7']*len(s)
    else:
        return ['background-color: #74b9ff']*len(s)

st.title('Meu primeiro App - GCI')

cidades = load_database()

dados, estatistica, outlier, zvalues, quadrante = st.tabs(['Dados', 'Estatística Descritiva', 'Outliers', 'Valores Padronizados', 'Quadrante'])

variaveis = ['area_territorial', 'populacao_estimada', 'densidade_demografica', 'pib_per_capita', 'idhm','carros']

with dados:
    if st.checkbox('Estado'):
        estado = st.selectbox('Selecione o Estado:', cidades['estado'].unique())
        st.dataframe(cidades[cidades['estado'] == estado])
    else:
        st.table(cidades)

with estatistica:
    variavel = st.selectbox('Selecione a variavel', variaveis)
    col1, col2, col3, col4 = st.columns([3,1,2,1])
    col1.altair_chart(alt.Chart(cidades).mark_bar().encode(x="municipio:O", y=variavel+':Q').properties(height=500))
    col2.dataframe(round(cidades[variavel].describe(),2))
    base = alt.Chart(cidades)
    bar = base.mark_bar().encode(x=alt.X(variavel+':Q', bin=True), y='count()')
    rule = base.mark_rule(color='red').encode(x='mean('+variavel+'):Q', size=alt.value(5))
    rule2 = base.mark_rule(color='green').encode(x='median('+variavel+'):Q', size=alt.value(5))
    col3.altair_chart(bar + rule + rule2)
    col4.altair_chart(alt.Chart(cidades).mark_boxplot().encode(y=variavel+':Q').properties(width=200))
with outlier:
    variavel = st.selectbox('Selecione a variavel para outliers', variaveis)
    cidades_var = cidades[['municipio', variavel]].copy()
    iqr = cidades_var[variavel].quantile(0.75) - cidades_var[variavel].quantile(0.25)
    out_min = cidades_var[variavel].quantile(0.25) - (1.5 * iqr)
    out_max = cidades_var[variavel].quantile(0.75) + (1.5 * iqr)
    limite_inferior = cidades_var[variavel].mean() - (1.96 * cidades_var[variavel].std() / np.sqrt(len(cidades_var)))
    limite_superior = cidades_var[variavel].mean() + (1.96 * cidades_var[variavel].std() / np.sqrt(len(cidades_var)))
    cidades_var['outlier_min'] = cidades_var[variavel].apply(lambda x : 1 if x < out_min else 0)
    cidades_var['outlier_max'] = cidades_var[variavel].apply(lambda x : 1 if x > out_max else 0)
    cidades_var['class_lc'] = cidades_var[variavel].apply(
        lambda x : 'abaixo' 
        if x < limite_inferior 
        else (
            'acima' 
            if x > limite_superior 
            else 'media'
        ) 
    )
    st.dataframe(cidades_var.style.apply(highlight_class_lc, axis=1))
    with st.expander('Média - Intervalo de Confiança'):
        col1, col2, col3 = st.columns(3)
        col1.metric('Média', round(cidades_var[variavel].mean(),2), "0")
        col2.metric('Limite Inferior', round(limite_inferior,2), round(limite_inferior - cidades_var[variavel].mean(),2))
        col3.metric('Limite Superior', round(limite_superior,2), round(limite_superior - cidades_var[variavel].mean(),2))
        st.altair_chart(alt.Chart(cidades_var).mark_bar().encode(x="municipio:O", y=variavel+':Q', color='class_lc:N').properties(height=400))        
    with st.expander('Outlier - Amplitude'):
        col1, col2, col3 = st.columns(3)
        col1.metric('Média', round(cidades_var[variavel].mean(),2), "0")
        col2.metric('Limite Inferior', round(out_min,2), round(out_min - cidades_var[variavel].mean(),2))
        col3.metric('Limite Superiorr', round(out_max,2), round(out_max - cidades_var[variavel].mean(),2))
        st.altair_chart(alt.Chart(cidades_var).mark_bar().encode(x="municipio:O", y=variavel+':Q', color='outlier_max:N').properties(height=400))
    with st.expander('Sem Outlier - Nova Média'):
        cidades_var_out = cidades_var[(cidades_var['outlier_max'] == 0) & (cidades_var['outlier_min'] == 0)].copy()
        limite_inferior = cidades_var_out[variavel].mean() - (1.96 * cidades_var_out[variavel].std() / np.sqrt(len(cidades_var_out)))
        limite_superior = cidades_var_out[variavel].mean() + (1.96 * cidades_var_out[variavel].std() / np.sqrt(len(cidades_var_out)))
        cidades_var_out['class_lc'] = cidades_var_out[variavel].apply(
            lambda x : 'abaixo' 
            if x < limite_inferior 
            else (
                'acima' 
                if x > limite_superior 
                else 'media'
            ) 
        )
        col1, col2, col3 = st.columns(3)
        col1.metric('Média', round(cidades_var_out[variavel].mean(),2), "0")
        col2.metric('Limite Inferior', round(limite_inferior,2), round(limite_inferior - cidades_var_out[variavel].mean(),2))
        col3.metric('Limite Superiorr', round(limite_superior,2), round(limite_superior - cidades_var_out[variavel].mean(),2))
        st.altair_chart(alt.Chart(cidades_var_out).mark_bar().encode(x="municipio:O", y=variavel+':Q', color='class_lc:N').properties(height=400))        
with zvalues:
    colunas = st.multiselect('colunas', variaveis)
    if len(colunas) > 0:
        sel = colunas
        sel.insert(0, "municipio")
        cidadesz = cidades[sel].copy()
        listaz = []
        for col in cidadesz.columns:
            if col != 'municipio':
                media = cidadesz[col].mean()
                dp = cidadesz[col].std()
                cidadesz['z_'+col] = cidadesz[col].apply(lambda x : (x - media) / dp)
                listaz.append('z_'+col)
        listaz.insert(0, "municipio")
        with st.expander('Dados'):
            st.dataframe(cidadesz.style.hide_index().background_gradient(cmap='Blues'))
        with st.expander('Gráfico'):
            graphz = pd.DataFrame()
            for zvalue in listaz:
                if zvalue != 'municipio':
                    for index, row in cidadesz.iterrows():
                        graphz = graphz.append({'municipio': row['municipio'], 'variable': zvalue, 'valor': row[zvalue]}, ignore_index=True)
            st.altair_chart(alt.Chart(graphz).mark_bar(opacity=0.5).encode(x='municipio:O', y='valor:Q', color='variable:N').properties(height=400))    
        with st.expander('Ranking'):
            if len(colunas) > 0:
                data = cidades[colunas]
                print(data)
                dataz = pd.DataFrame()
                for col in data.columns:
                    if col != 'municipio':
                        media = cidades[col].mean()
                        dp = cidades[col].std()
                        dataz[col] = cidades[col].apply(lambda x: (x - media) / dp)
                dataz['total'] = dataz.sum(
                    axis=1,
                    skipna=True
                )
                dataz['ranking'] = dataz['total'].rank(ascending=False)
                iqr = dataz['total'].quantile(0.75) - dataz['total'].quantile(0.25)
                out_min = dataz['total'].quantile(0.25) - (1.5 * iqr)
                out_max = dataz['total'].quantile(0.75) + (1.5 * iqr)
                erro = 1.96 * dataz['total'].std() / np.sqrt(len(data))
                li = dataz['total'].mean() - erro
                ls = dataz['total'].mean() + erro
                dataz['zscore'] = (dataz['total'] - dataz['total'].mean()) / dataz['total'].std()
                dataz['stars'] = round(dataz['zscore'], 0) + 3
                dataz['outlier_min'] = dataz['total'].apply(
                    lambda x: 1 if x < out_min
                    else 0
                )
                dataz['outlier_max'] = dataz['total'].apply(
                    lambda x: 1 if x > out_max
                    else 0
                )
                media = dataz['total'].mean()
                dataz['class_media'] = dataz['total'].apply(
                    lambda x: 'abaixo' if x < media
                    else 'acima'
                )
                dataz['class_lc'] = dataz['total'].apply(
                    lambda x: 'abaixo' if x < li
                    else (
                        'acima' if x > ls
                        else 'media'
                    )
                )
                datac = cidades[['municipio','estado']].copy()
                datac = datac.merge(dataz, left_index=True, right_index=True)
                data_sort = datac.sort_values(by='ranking')
                st.table(data_sort.style.hide_index().background_gradient(cmap='Blues'))