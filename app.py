import streamlit as st
import pandas as pd
from models import load_dataset, train_models
from plots import plot_confusion_matrix, plot_feature_importance, plot_roc
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

st.title("Wine Classification - Machine Learning Comparison")

st.sidebar.header("Hyperparameters")

max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 50)

X, y, wine = load_dataset()

results, X_test, y_test = train_models(max_depth, n_estimators)

st.header("Model Comparison")

data = []

for name, res in results.items():

    data.append({
        "Model": name,
        "Accuracy": res["accuracy"],
        "Std": res["std"]
    })

df = pd.DataFrame(data)

st.dataframe(df)

st.bar_chart(df.set_index("Model")["Accuracy"])


model_name = st.selectbox(
    "Select model for analysis",
    list(results.keys())
)

model = results[model_name]["model"]

st.subheader("Confusion Matrix")

fig = plot_confusion_matrix(results[model_name]["cm"])

st.pyplot(fig)

st.subheader("ROC Curve")

fig = plot_roc(model, X_test, y_test)

st.pyplot(fig)


if model_name == "Decision Tree":

    st.subheader("Feature Importance")

    fig = plot_feature_importance(model, wine.feature_names)

    st.pyplot(fig)

    st.subheader("Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(12,8))

    plot_tree(
        model,
        feature_names=wine.feature_names,
        class_names=wine.target_names,
        filled=True
    )

    st.pyplot(fig)
