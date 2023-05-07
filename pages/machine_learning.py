from sklearn.metrics import classification_report, mean_squared_error
import streamlit as st
import os 
import json
import pandas as pd
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

def standardization(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
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

def knn_models(X_train, X_test, y_train, y_test, n_neighbors=5):
    from sklearn.neighbors import KNeighborsClassifier

    # Instantiate the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model
    knn.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test)

    # show classification report in streamlit
    st.subheader("Avaliação KNN")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

def kmeans_models(X_train, X_test, y_train, y_test, n_clusters=5):
    from sklearn.cluster import KMeans

    # Instantiate the model
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit the model
    kmeans.fit(X_train, y_train)

    # Predict on test set
    y_pred = kmeans.predict(X_test)

    st.subheader("Avaliação")
    st.write("Erro quadrático médio", mean_squared_error(y_test, y_pred))

def random_forest_classifier(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

    # Create a based model
    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Avaliação Random Forest")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)
    

def random_forest_regressor(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    # Create a based model
    rf = RandomForestRegressor()

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Avaliação Random Forest")
    st.write("Erro quadrático médio", mean_squared_error(y_test, y_pred))

def svm_classifier(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC

    # Instantiate the model
    svm = SVC()

    # Fit the model
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    st.subheader("Avaliação SVM")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

def svm_regressor(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVR

    # Instantiate the model
    svm = SVR()

    # Fit the model
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    st.subheader("Avaliação SVM")
    st.write("Erro quadrático médio", mean_squared_error(y_test, y_pred))

def logistic_regression_models(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression

    # Instantiate the model
    logreg = LogisticRegression()

    # Fit the model
    logreg.fit(X_train, y_train)

    # Predict on test set
    y_pred = logreg.predict(X_test)

    st.subheader("Avaliação Regressão Logística")
    st.write("Erro quadrático médio", mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    options = load_settings()
    st.title("Machine Learning")
    df = None

    if options['file'] is False:
        file = st.file_uploader("Upload do arquivo CSV", type="csv")
        if file is not None:
            df = pd.read_csv(file)
            df.to_csv('data.csv', index=False)
            options['file'] = True
    else:
        df = pd.read_csv('data.csv')
    if df is not None:
        with st.form(key='my_form'):
            target = st.selectbox('Selecione a coluna alvo', df.columns)
            submited = st.form_submit_button(label='ok')
        
        if submited:
            X_train, X_test, y_train, y_test = split_data_supervised(df, target)
            if options['approach'] == 'Classificação':
                if options['processing_super'] != 'Sem processamento':
                    processing_super = options['processing_super']
                    if processing_super == 'Normalização':
                        X_train, X_test = normalization(X_train, X_test)
                    if processing_super == 'Padronização':
                        X_train, X_test = standardization(X_train, X_test)
                if options['knn']:
                    knn_models(X_train, X_test, y_train, y_test)
                if options['random_forest_classifier']:
                    random_forest_classifier(X_train, X_test, y_train, y_test)
                if options['svm_classifier']:
                    svm_classifier(X_train, X_test, y_train, y_test)
        
            if options['approach'] == 'Regressão':
                #X_train, X_test = split_data_unsupervised(df)
                if options['processing_unsuper'] != 'Sem processamento':
                    processing_unsuper = options['processing_unsuper']
                    if processing_unsuper == 'PCA':
                        X_train, X_test , explained_variance = reduce_data_with_pca(X_train, X_test)
                    if processing_unsuper == 'LDA':
                        X_train, X_test, explained_variance = reduce_data_with_lda(X_train, X_test)
                
                # if options['kmeans']:
                #     kmeans_models(X_train, X_test, y_train, y_test)
                if options['random_forest_regressor']:
                    random_forest_regressor(X_train, X_test, y_train, y_test)
                if options['svm_regressor']:
                    svm_regressor(X_train, X_test, y_train, y_test)
                if options['logistic_regression']:
                    logistic_regression_models(X_train, X_test, y_train, y_test)
        if st.button('Retornar ao inicio'):
            if options['viz']:
                if options['pross']:
                    switch_page("data_processing")
                switch_page("plotting")
            elif options['ml']:
                if options['pross']:
                    switch_page("data_processing")
                switch_page("machine_learning")