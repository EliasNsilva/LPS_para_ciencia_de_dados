import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from persist import load_widget_state
from ml_models import train_model, randon_florest_models, svm_models, linear_models

st.set_page_config(page_title="LPS")

def main():
    if "page" not in st.session_state:
        st.session_state.update({
            # Default page.
            "page": "settings",

            # Default widget values.
            "line_plot": False,
            "bar_plot": False,
            "checkbox": False,
            "scatter_plot": False,
            "pair_plot": False,
            "knn": False,
            "svm": False,
            "random_forest": False,
            "logistic_regression": False,
            })

    if "remove_page" not in st.session_state:
        st.session_state.remove_page = False
    
    if st.session_state.remove_page:
        del PAGES['settings']
    
    page = st.sidebar.selectbox("Options", options = tuple(PAGES.keys()), format_func=str.capitalize)
    PAGES[page]()

def first_page():
    with st.container():
        st.subheader('Escolha os tipos de gráficos')
        line_plot = st.checkbox('Gráfico de linha', key="line_plot")
        bar_plot = st.checkbox('Gráfico de barras', key="bar_plot")
        scatter_plot = st.checkbox('Gráfico de dispersão', key="scatter_plot")
        pair_plot = st.checkbox('Gráfico de pares', key="checkbox")
        
        st.subheader('Escolha os modelos de aprendizado de máquina')
        knn = st.checkbox('KNN', key="knn")
        svm = st.checkbox('SVM', key="svm")
        random_forest = st.checkbox('Random Forest', key="random_forest")
        logistic_regression = st.checkbox('Logistic Regression', key="logistic_regression")
    
    if st.button("Next"):
        st.session_state.remove_page = True

def second_page():
    file = st.file_uploader("Upload do arquivo CSV", type="csv")

    # Verifica se o usuário fez o upload do arquivo
    if file is not None:
        df = pd.read_csv(file)
        
        st.subheader('Tabela de dados')
        st.dataframe(df)

PAGES = {
    "settings": first_page,
    "data": second_page,
}

if __name__ == "__main__":
    load_widget_state()
    main()
