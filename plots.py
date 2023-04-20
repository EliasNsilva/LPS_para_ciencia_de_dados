import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def line_plot(df):
    x = st.selectbox("Escolha uma coluna para o eixo x do grafico de linha", df.columns)
    y = st.multiselect("Escolha as colunas para o eixo y do grafico de linha", df.columns)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], label=y)
    plt.legend()
    st.pyplot(fig)

def bar_plot(df):
    x = st.selectbox("Escolha uma coluna para o eixo x do gráfico de barras", df.columns)
    #y = st.multiselect("Escolha as colunas para o eixo y do gráfico de barras", df.columns)
    #fig = plt.figure(figsize=(10, 6))
    #plt.bar(df[x], df[y], label=y)

    #plot = df[x].value_counts().plot(kind='bar', figsize=(15, 8))
    #plt.legend()
    #fig = plot.get_figure()
    #st.pyplot(fig)
    
    st.bar_chart(df[x].value_counts())

def scatter_plot(df):
    x = st.selectbox("Escolha uma coluna para o eixo x do gráfico de dispersão", df.columns)
    y = st.selectbox("Escolha uma coluna para o eixo y do gráfico de dispersão", df.columns)
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], label=y)
    st.pyplot(fig)

def pair_plot(df):
    hue = st.selectbox("Escolha a coluna para o hue", df.columns)
    fig = plt.figure(figsize=(10, 6))
    sns.pairplot(df, hue=hue)
    st.pyplot(fig)