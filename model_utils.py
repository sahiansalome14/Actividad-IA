from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import numpy as np

def load_data():
    data = load_wine()
    X = data.data
    y = data.target
    return X, y, data.target_names


def train_decision_tree(max_depth):

    X, y, _ = load_data()

    model = DecisionTreeClassifier(max_depth=max_depth)

    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean(), scores.std()


def train_bagging(max_depth, n_estimators):

    X, y, _ = load_data()

    base_model = DecisionTreeClassifier(max_depth=max_depth)

    model = BaggingClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        bootstrap=True
    )

    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean(), scores.std()


def train_boosting(n_estimators):

    X, y, _ = load_data()

    model = AdaBoostClassifier(
        n_estimators=n_estimators
    )

    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean(), scores.std()


def roc_metrics():

    X, y, _ = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=3)

    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=[0,1,2])

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc
