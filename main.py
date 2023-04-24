import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from persist import load_widget_state
from plots import *
from data_processing import *
from ml_models import *

st.set_page_config(page_title="LPS")

def main():
    if "page" not in st.session_state:
        st.session_state.update({
            # Default page.
            "page": "settings",
            })

    if "remove_page" not in st.session_state:
        st.session_state.remove_page = False

    if "line_plot" not in st.session_state:
        st.session_state["line_plot"] = False
    
    if "bar_plot" not in st.session_state:
        st.session_state["bar_plot"] = False

    if "scatter_plot" not in st.session_state:
        st.session_state["scatter_plot"] = False
    
    if "pair_plot" not in st.session_state:
        st.session_state["pair_plot"] = False

    if "knn" not in st.session_state:
        st.session_state["knn"] = False
    
    if "svm" not in st.session_state:
        st.session_state["svm"] = False
    
    if "random_forest" not in st.session_state:
        st.session_state["random_forest"] = False
    
    if "logistic_regression" not in st.session_state:
        st.session_state["logistic_regression"] = False

    if st.session_state.remove_page:
        del PAGES['settings']
    
    page = st.sidebar.selectbox("Menu", options = tuple(PAGES.keys()), format_func=str.capitalize)
    PAGES[page]()

def remove_page():
    st.balloons()
    st.session_state.remove_page = True

def first_page():
    with st.container():
        st.subheader('Escolha os módulos que deseja utilizar')
        pross = st.checkbox('Processamento de dados', key="pross")
        viz = st.checkbox('Vizualização de dados', key="viz")
        ml = st.checkbox('Aprendizado de máquina', key="ml")

        if pross:
            pass
        if viz:
            st.subheader('Escolha os tipos de gráficos')
            st.checkbox('Gráfico de linha', key="line_plot")
            st.checkbox('Gráfico de barras', key="bar_plot")
            st.checkbox('Gráfico de dispersão', key="scatter_plot")
            st.checkbox('Gráfico de pares', key="checkbox")
        if ml:
            st.subheader('Escolha os modelos de aprendizado de máquina')
            st.checkbox('KNN', key="knn")
            st.checkbox('SVM', key="svm")
            st.checkbox('Random Forest', key="random_forest")
            st.checkbox('Logistic Regression', key="logistic_regression")
        
    st.button("Next", on_click=remove_page)

def second_page():
    file = st.file_uploader("Upload do arquivo CSV", type="csv")

    st.session_state['pross'] = st.session_state.get('pross', False)
    st.session_state['viz'] = st.session_state.get('viz', False)
    st.session_state['ml'] = st.session_state.get('ml', False)

    st.session_state["line_plot"] = st.session_state.get("line_plot", False)
    st.session_state["bar_plot"] = st.session_state.get("bar_plot", False)
    st.session_state["scatter_plot"] = st.session_state.get("scatter_plot", False)
    st.session_state["pair_plot"] = st.session_state.get("pair_plot", False)
    
    st.session_state["knn"] = st.session_state.get("knn", False)
    st.session_state["svm"] = st.session_state.get("svm", False)
    st.session_state["random_forest"] = st.session_state.get("random_forest", False)
    st.session_state["logistic_regression"] = st.session_state.get("logistic_regression", False)


    # # Verifica se o usuário fez o upload do arquivo
    if file is not None:
        df = pd.read_csv(file)    
        st.subheader('Tabela dos dados')
        st.dataframe(df)

        if st.session_state['viz']:
            if st.session_state['line_plot']:
                st.subheader('Grafico de linha')
                line_plot(df)

            if st.session_state['bar_plot']:
                st.subheader('Grafico de barras')
                bar_plot(df)

            if st.session_state['scatter_plot']:
                st.subheader('Grafico de dispersão')
                scatter_plot(df)

            if st.session_state['pair_plot']:
                st.subheader('Grafico de pares')
                pair_plot(df)
        
        if st.session_state['ml']:
            with st.form(key='form_ml'):
                st.subheader('Aprendizado de máquina')
                target = st.selectbox('Selecione a variável alvo', df.columns)
                test_size = st.number_input('Tamanho do conjunto de teste', min_value=0.0, max_value=1.0, value=0.2)
                X_train, X_test, y_train, y_test = split_data(df, target, test_size)
                submit = st.form_submit_button(label='Submit')
                
                if submit:
                    if st.session_state['knn']:
                        st.subheader('KNN')
                        n_neighbors = st.number_input('Número de vizinhos', min_value=1, max_value=10, value=5)
                        knn_models(X_train, X_test, y_train, y_test, n_neighbors)
                    if st.session_state['svm']:
                        st.subheader('SVM')
                        svm_models(X_train, X_test, y_train, y_test)
                    if st.session_state['random_forest']:
                        st.subheader('Random Forest')
                        random_forest_models(X_train, X_test, y_train, y_test)
                    if st.session_state['logistic_regression']:
                        st.subheader('Logistic Regression')
                        logistic_regression_models(X_train, X_test, y_train, y_test)

def third_page():
    st.subheader("Processamento de Dados - WIP")
    
def fourth_page():
    st.subheader("ML - WIP")
    if st.session_state['knn']:
        st.subheader("KNN - WIP")

PAGES = {
    "settings": first_page,
    "data": second_page,
    "data processing": third_page,
    "machine learning": fourth_page,
}

if __name__ == "__main__":
    load_widget_state()
    main()
