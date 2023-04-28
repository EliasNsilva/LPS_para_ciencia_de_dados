import json
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit as st
import json

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

options = {}

def send_button():
    if 'button' not in st.session_state:
        st.session_state.button = False

def first_page():
    with st.container():
        st.subheader('Escolha os módulos que deseja utilizar')
        viz = st.checkbox('Visualização de dados', key="viz")
        pross = st.checkbox('Processamento de dados', key="pross")
        ml = st.checkbox('Aprendizado de máquina', key="ml")

        if pross:
            options['pross'] = pross
            st.subheader('Escolha processamento de dados')
            missing_data = st.checkbox('Remoção de dados faltantes', key="missing_data") 
            options['missing_data'] = missing_data

            outliers = st.checkbox('Remoção de outliers', key="outliers")
            options['outliers'] = outliers

            categorical_data = st.checkbox('Transformação de dados categóricos', key="categorical_data")
            options['categorical_data'] = categorical_data

        if viz:
            options['viz'] = viz
            st.subheader('Escolha os tipos de gráficos')
            line_plot = st.checkbox('Gráfico de linha', key="line_plot")
            options['line_plot'] = line_plot

            bar_plot = st.checkbox('Gráfico de barras', key="bar_plot")
            options['bar_plot'] = bar_plot

            scatter_plot = st.checkbox('Gráfico de dispersão', key="scatter_plot")
            options['scatter_plot'] = scatter_plot

            checkbox = st.checkbox('Gráfico de pares', key="checkbox")
            options['pair_plot'] = checkbox

        if ml:
            options['ml'] = ml
            supervised = st.checkbox('Abordagens supervisionadas?')
            options['supervised'] = supervised

            if supervised:
                st.subheader('Escolha os tipos de processamento de dados')
                normalization = st.checkbox('Normalização', key="normalization")
                options['normalization'] = normalization

                standardization = st.checkbox('Padronização', key="standardization")
                options['standardization'] = standardization

                st.subheader('Escolha os modelos de aprendizado de máquina')
                knn = st.checkbox('KNN', key="knn")
                options['knn'] = knn

                svm = st.checkbox('SVM', key="svm")
                options['svm'] = svm

                random_forest = st.checkbox('Random Forest', key="random_forest")
                options['random_forest'] = random_forest

            else:
                st.subheader('Escolha os tipos de processamento de dados')
                dimensionality_reduction_PCA = st.checkbox('Redução de dimensionalidade PCA', key='dimensionality_reduction_PCA')
                options['dimensionality_reduction_PCA'] = dimensionality_reduction_PCA

                dimensionality_reduction_LDA = st.checkbox('Redução de dimensionalidade LDA', key='dimensionality_reduction_LDA')
                options['dimensionality_reduction_LDA'] = dimensionality_reduction_LDA

                st.subheader('Escolha os modelos de aprendizado de máquina')
                logistic_regression = st.checkbox('Logistic Regression', key="logistic_regression")
                options['logistic_regression'] = logistic_regression

    with open('settings.json', 'w') as f:
        json.dump(options, f)

    if st.button('Selecionar'):
        st.session_state.button = True 

if __name__ == "__main__":
    send_button()
    first_page()

    if st.session_state.button:
        if options['viz']:
            switch_page("plotting")

