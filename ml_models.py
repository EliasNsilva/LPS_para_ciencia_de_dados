from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(data, target, test_size=0.2):
    # Split data into train and test sets
    X = data.drop(target, axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def randon_florest_models(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    # Create a based model
    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def svm_models(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report

    # Instantiate the model
    svm = SVC()

    # Fit the model
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

def linear_models(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    # Instantiate the model
    logreg = LogisticRegression()

    # Fit the model
    logreg.fit(X_train, y_train)

    # Predict on test set
    y_pred = logreg.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))