from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd


def load_dataset():

    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    return X, y, wine


def train_models(max_depth, n_estimators):

    X, y, _ = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    dt = DecisionTreeClassifier(max_depth=max_depth)

    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators
    )

    boost = AdaBoostClassifier(
        n_estimators=n_estimators
    )

    models = {
        "Decision Tree": dt,
        "Bagging": bag,
        "Boosting": boost
    }

    results = {}

    for name, model in models.items():

        scores = cross_val_score(model, X, y, cv=5)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": scores.mean(),
            "std": scores.std(),
            "cm": cm
        }

    return results, X_test, y_test
