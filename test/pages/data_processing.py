import os
import json
import tempfile
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def send_button():
    if 'button_pross' not in st.session_state:
        st.session_state.button_pross = False

def split_data_supervised(data, target, test_size=0.2):
    # Split data into train and test sets
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test

def split_data_unsupervised(data, test_size=0.2):
    # Split data into train and test sets
    X_train, X_test = train_test_split(data, test_size=test_size, random_state=0)
    return X_train, X_test

def remove_missing_data(data):
    # Missing data
    st.subheader("Dados faltantes")
    st.dataframe(data.isnull().sum())
    remove =  st.checkbox("Remover dados faltantes?")
    if remove:
        data.dropna(inplace=True)
    
    return data

def remove_outliers(data):
    fig = plt.figure(figsize=(20, 10))
    sns.boxplot(data=data, orient="h", palette="Set2")
    st.pyplot(fig)

def convert_with_get_dummies(data, columns):
    # Convert categorical variable into dummy/indicator variables
    data = pd.get_dummies(data, columns=columns)
    st.subheader("Dados convertidos com get_dummies")
    st.dataframe(data)
    return data

def standardization(X_train, X_test):
    # Scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def normalization(X_train, X_test):
    # Normalize data
    from sklearn.preprocessing import Normalizer

    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

def reduce_data_with_pca(X_train, X_test, n_components=2):
    # Reduce data
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    return X_train, X_test, explained_variance

def reduce_data_with_lda(X_train, X_test, n_components=2):
    # Reduce data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA(n_components=n_components)
    X_train = lda.fit_transform(X_train)
    X_test = lda.transform(X_test)
    explained_variance = lda.explained_variance_ratio_
    return X_train, X_test, explained_variance

if __name__ == "__main__":
    send_button()
    options = load_settings()

    file = st.file_uploader("Upload do arquivo CSV", type="csv")

    if file is not None:
        df = pd.read_csv(file)
        #Pegando o nome do arquivo
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            file_path = temp_file.name

        filename = os.path.basename(file_path)

        if options['missing_data']:
            df = remove_missing_data(df)
        if options['outliers']:
            remove_outliers(df)
        if options['categorical_data']:
            df = convert_with_get_dummies(df)

    if st.button('Aplicar'):
        st.session_state.button_pross = True 
    
    if st.session_state.button_pross:
        df.to_csv(os.getcwd() + '/' + filename + '.csv', index=False)
        options['filename'] = filename + '.csv'

        with open('settings.json', 'w') as f:
            json.dump(options, f)

        if options['viz']:
            switch_page("plotting")
        elif options['ml']:
            switch_page("machine_learning")