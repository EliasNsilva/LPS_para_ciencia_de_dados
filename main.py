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
        pross = st.checkbox('Processamento de dados', key="pross")
        viz = st.checkbox('Visualização de dados', key="viz")
        ml = st.checkbox('Aprendizado de máquina', key="ml")

        options['pross'] = pross
        options['viz'] = viz
        options['ml'] = ml
        options['file'] = False

        if pross:
            st.subheader('Escolha processamento de dados')
            missing_data = st.checkbox('Remoção de dados faltantes', key="missing_data") 
            options['missing_data'] = missing_data

            outliers = st.checkbox('Remoção de outliers', key="outliers")
            options['outliers'] = outliers

            remove_column = st.checkbox('Remover coluna do dataframe', key="remove_column")
            options['remove_column'] = remove_column

            categorical_data = st.checkbox('Transformação de dados categóricos usando get dummies', key="categorical_data")
            options['categorical_data'] = categorical_data

            encode_data = st.checkbox('Transformação de dados categóricos usando Label Encoder', key="encode_data")
            options['encode_data'] = encode_data

        if viz:
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
            approach = st.selectbox('Escolha o tipo de abordagem', ['Classificação', 'Regressão'], key='approach')
            options['approach'] = approach

            if approach == 'Classificação':
                st.subheader('Escolha os tipos de processamento de dados')
                options['processing_super'] = st.selectbox('Escolha o tipo de abordagem', ['Sem processamento', 'Normalização', 'Padronização'], key='processing_unsuper')

                st.subheader('Escolha os modelos de aprendizado de máquina')
                knn = st.checkbox('KNN', key="knn")
                options['knn'] = knn

                svm = st.checkbox('SVM', key="svm")
                options['svm_classifier'] = svm

                random_forest = st.checkbox('Random Forest', key="random_forest")
                options['random_forest_classifier'] = random_forest

            elif approach == 'Regressão':
                st.subheader('Escolha os tipos de processamento de dados')
                
                options['processing_unsuper'] = st.selectbox('Escolha o tipo de abordagem', ['Sem processamento', 'PCA', 'LDA'], key='processing_unsuper')

                st.subheader('Escolha os modelos de aprendizado de máquina')
                logistic_regression = st.checkbox('Logistic Regression', key="logistic_regression")
                options['logistic_regression'] = logistic_regression

                # kmeans = st.checkbox('K-Means', key="kmeans")
                # options['kmeans'] = kmeans

                svm = st.checkbox('SVM', key="svm_regressor")
                options['svm_regressor'] = svm

                random_forest = st.checkbox('Random Forest', key="random_forest_regressor")
                options['random_forest_regressor'] = random_forest

    with open('settings.json', 'w') as f:
        json.dump(options, f)

    if st.button('Selecionar'):
        st.session_state.button = True 

if __name__ == "__main__":
    send_button()
    first_page()

    if st.session_state.button:
        if options['viz']:
            if options['pross']:
                switch_page("data_processing")
            switch_page("plotting")
        elif options['ml']:
            if options['pross']:
                switch_page("data_processing")
            switch_page("machine_learning")
    

