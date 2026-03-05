import streamlit as st
import matplotlib.pyplot as plt
from model_utils import train_decision_tree, train_bagging, train_boosting, roc_metrics

st.title("Clasificación Wine Dataset")

st.sidebar.header("Parámetros")

max_depth = st.sidebar.slider("Profundidad del árbol", 1, 10, 3)
n_estimators = st.sidebar.slider("Número de estimadores", 10, 200, 50)

st.header("Resultados")

st.subheader("Decision Tree")

mean_acc, std_acc = train_decision_tree(max_depth)

st.write("Accuracy promedio:", mean_acc)
st.write("Desviación:", std_acc)

st.subheader("Bagging")

mean_acc, std_acc = train_bagging(max_depth, n_estimators)

st.write("Accuracy promedio:", mean_acc)
st.write("Desviación:", std_acc)

st.subheader("Boosting")

mean_acc, std_acc = train_boosting(n_estimators)

st.write("Accuracy promedio:", mean_acc)
st.write("Desviación:", std_acc)

st.subheader("Curva ROC")

fpr, tpr, roc_auc = roc_metrics()

fig, ax = plt.subplots()

for i in range(3):
    ax.plot(fpr[i], tpr[i], label=f'Clase {i} AUC={roc_auc[i]:.2f}')

ax.plot([0,1],[0,1],'--')

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

st.pyplot(fig)
