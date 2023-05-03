import os
import json
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="LPS", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

def load_settings():
    if os.path.exists('settings.json'):
        with open('settings.json', 'r') as f:
            return json.load(f)
    else:
        return {}

def line_plot(df):
    x = st.selectbox("Escolha uma coluna para o eixo x do grafico de linha", df.columns)
    y = st.multiselect("Escolha as colunas para o eixo y do grafico de linha", df.columns)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], label=y)
    plt.legend()
    st.pyplot(fig)

def bar_plot(df):
    x = st.selectbox("Escolha uma coluna para o eixo x do gráfico de barras", df.columns)
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


if __name__ == "__main__":
    options = load_settings()

    file = st.file_uploader("Upload do arquivo CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        st.subheader('Tabela dos dados')
        st.dataframe(df)

        if options['line_plot']:
            line_plot(df)
        if options['bar_plot']:
            bar_plot(df)
        if options['scatter_plot']:
            scatter_plot(df)
        if options['pair_plot']:
            pair_plot(df)