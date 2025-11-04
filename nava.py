
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
ruta = "dataset_flores_ruido (1).xlsx"
data = pd.read_excel(ruta)
print("== Datos cargados del archivo == ")
print(data.head())
X = data[["Largo_Petalo", "Ancho_Petalo"]]
y = data["Tipo_Flor"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\n=== Métricas del modelo ===")
print("Precisión:", round(accuracy_score(y_test, y_pred), 3))
print("Sensibilidad (Recall):", round(recall_score(y_test, y_pred, average='macro'), 3))
print("F1-Score:", round(f1_score(y_test, y_pred, average='macro'), 3))
print("\nReporte completo:")
print(classification_report(y_test, y_pred))
patrones = pd.DataFrame({
    "Largo_Petalo": [5.1235, 4.9842, 6.1213, 2.3123],
    "Ancho_Petalo": [6.8832, 7.8921, 4.1235, 8.1231]
}, index=["P1", "P2", "P3", "P4"])
predicciones = knn.predict(patrones)
patrones["Tipo_Flor_Predicho"] = predicciones
print("\n== Clasificación de los patrones P1–P4 ==")
print(patrones)
plt.figure(figsize=(8,6))
for clase in y.unique():
    datos_clase = data[data["Tipo_Flor"] == clase]
    plt.scatter(datos_clase["Largo_Petalo"], datos_clase["Ancho_Petalo"], label=clase, alpha=0.7)
plt.scatter(patrones["Largo_Petalo"], patrones["Ancho_Petalo"], 
            c='black', marker='x', s=100, label='Patrones P1–P4')
plt.title("Diagrama de dispersión - KNN Flores")
plt.xlabel("Largo_Petalo")
plt.ylabel("Ancho_Petalo")
plt.legend()
plt.grid(True)
plt.show()