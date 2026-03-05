import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(cm):

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig


def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    fig, ax = plt.subplots()

    sns.barplot(x=importance, y=feature_names)

    ax.set_title("Feature Importance")

    return fig


def plot_roc(model, X_test, y_test):

    y_score = model.predict_proba(X_test)

    y_bin = label_binarize(y_test, classes=[0,1,2])

    fig, ax = plt.subplots()

    for i in range(3):

        fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])

        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"Clase {i} AUC={roc_auc:.2f}")

    ax.plot([0,1],[0,1],'--')

    ax.set_title("ROC Curve")

    ax.legend()

    return fig
