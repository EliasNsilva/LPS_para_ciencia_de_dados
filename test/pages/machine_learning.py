from sklearn.metrics import classification_report
import streamlit as st
from sklearn.linear_model import LogisticRegression
import os 
import json

from data_processing import *

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

def knn_models(X_train, X_test, y_train, y_test, n_neighbors=5):
    from sklearn.neighbors import KNeighborsClassifier

    # Instantiate the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model
    knn.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))
    # show classification report in streamlit
    st.subheader("Classification report")
    st.write(classification_report(y_test, y_pred))

def kmeans_models(X_train, X_test, y_train, y_test, n_clusters=5):
    from sklearn.cluster import KMeans

    # Instantiate the model
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit the model
    kmeans.fit(X_train, y_train)

    # Predict on test set
    y_pred = kmeans.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def random_forest_classifier(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

    # Create a based model
    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def random_forest_regressor(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    # Create a based model
    rf = RandomForestRegressor()

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def svm_classifier(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC

    # Instantiate the model
    svm = SVC()

    # Fit the model
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def svm_regressor(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVR

    # Instantiate the model
    svm = SVR()

    # Fit the model
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def logistic_regression_models(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression

    # Instantiate the model
    logreg = LogisticRegression()

    # Fit the model
    logreg.fit(X_train, y_train)

    # Predict on test set
    y_pred = logreg.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    options = load_settings()
    df = pd.read_csv(options['filename'])

    if options['approach'] == 'Supervisionado':
        with st.form(key='my_form'):
            target = st.selectbox('Selecione a coluna alvo', df.columns)
            submited = st.form_submit_button(label='ok')
        if submited:
            X_train, X_test, y_train, y_test = split_data_supervised(df, options['target'])
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
    if options['approach'] == 'Não supervisionado':
        X_train, X_test = split_data_unsupervised(df)
        if options['processing_unsuper'] != 'Sem processamento':
            processing_unsuper = options['processing_unsuper']
            if processing_unsuper == 'PCA':
                X_train, X_test , explained_variance = reduce_data_with_pca(X_train, X_test)
            if processing_unsuper == 'LDA':
                X_train, X_test, explained_variance = reduce_data_with_lda(X_train, X_test)
        if options['kmeans']:
            kmeans_models(X_train, X_test, y_train, y_test)
        if options['random_forest_regressor']:
            random_forest_regressor(X_train, X_test, y_train, y_test)
        if options['svm_regressor']:
            svm_regressor(X_train, X_test, y_train, y_test)
        if options['logistic_regression']:
            logistic_regression_models(X_train, X_test, y_train, y_test)