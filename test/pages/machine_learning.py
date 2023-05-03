import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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