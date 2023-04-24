import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_data(data, target, test_size=0.2):
    # Split data into train and test sets
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test

def convert_with_get_dummies(data, columns):
    # Convert categorical variable into dummy/indicator variables
    data = pd.get_dummies(data, columns=columns)
    st.subheader("Dados convertidos com get_dummies")
    st.dataframe(data)
    return data

def scale_data(X_train, X_test):
    # Scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def normalize_data(X_train, X_test):
    # Normalize data
    from sklearn.preprocessing import Normalizer

    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

def reduce_data(X_train, X_test, y_train, y_test, n_components=2):
    # Reduce data
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    return X_train, X_test, y_train, y_test, explained_variance

def reduce_data_with_lda(X_train, X_test, y_train, y_test, n_components=2):
    # Reduce data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA(n_components=n_components)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    explained_variance = lda.explained_variance_ratio_
    return X_train, X_test, y_train, y_test, explained_variance