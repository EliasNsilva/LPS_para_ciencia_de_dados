import os
import json
import tempfile
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page

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

def save_file(df):
    df.to_csv('data.csv', index=False)

def remove_missing_data():
    data = pd.read_csv('data.csv')
    # Missing data
    st.subheader("Dados faltantes")
    st.dataframe(data.isnull().sum())
    remove =  st.checkbox("Remover dados faltantes?")
    if remove:
        data.dropna(inplace=True)
        save_file(data)


def remove_outliers():
    data = pd.read_csv('data.csv')

    fig = plt.figure(figsize=(20, 10))
    sns.boxplot(data=data, orient="h", palette="Set2")
    st.pyplot(fig)
    
    if st.checkbox("Remover outliers?"):
        # Remove outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        data = data[(data > lower_limit) & (data < upper_limit)]

        st.write("Dados sem outliers")
        fig2 = plt.figure(figsize=(20, 10))
        sns.boxplot(data=data, orient="h", palette="Set2")
        st.pyplot(fig2)
        
        save_file(data)


def convert_with_get_dummies():
    data = pd.read_csv('data.csv')
    # Convert categorical variable into dummy/indicator variables
    with st.form(key='form_dummies'):
        columns = st.multiselect("Selecione as colunas para converter com get_dummies", list(data.columns))
        submit = st.form_submit_button("Converter com get_dummies")
    if submit:
        data = pd.get_dummies(data, columns=columns)
        st.subheader("Dados convertidos com get_dummies")
        st.dataframe(data)
        save_file(data)

def convert_with_label_encoder():
    data = pd.read_csv('data.csv')
    # Convert categorical variable into dummy/indicator variables
    with st.form(key='form_encoder'):
        columns = st.multiselect("Selecione as colunas para converter com LabelEncoder", list(data.columns))
        submit = st.form_submit_button("Converter com LabelEncoder")
    if submit:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
        st.subheader("Dados convertidos com LabelEncoder")
        st.dataframe(data)
        save_file(data)

def remove_columns():
    df = pd.read_csv('data.csv')

    columns = list(df.columns)
    columns_to_remove = []

    if len(columns) == 0:
        st.write("Não há mais colunas para remover")
        return df
    
    with st.form(key= 'form'):
        columns_to_remove = st.multiselect(
            f"Selecione uma coluna para remover ({len(columns)} restantes)", 
            options=columns)
        remove = st.form_submit_button("Remover")

    if remove:            
        df.drop(columns=columns_to_remove, inplace=True)
        st.write('Dataset após remoção das colunas')
        st.write(df)
        save_file(df)

if __name__ == "__main__":
    options = load_settings()
    st.header('Processamento de dados')
    file = st.file_uploader("Upload do arquivo CSV", type="csv")
    if file is not None:
        st.write("Arquivo carregado com sucesso!")
        df = pd.read_csv(file)
        df.to_csv('data.csv', index=False)
        st.dataframe(df)

        if options['missing_data']:
            remove_missing_data()
        if options['outliers']:
            remove_outliers()
        if options['categorical_data']:
            convert_with_get_dummies()
        if options['remove_column']:
            remove_columns()
        if options['encode_data']:
            convert_with_label_encoder()

    if st.button('Concluir'):
        options['file'] = True
        
        with open('settings.json', 'w') as f:
            json.dump(options, f)

        if options['viz']:
            switch_page("plotting")
        elif options['ml']:
            switch_page("machine_learning")
        else:
            st.write('Operação concluída com sucesso!')