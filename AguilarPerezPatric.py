import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

archivo = "flores.xlsx"

datos = pd.read_excel(archivo)

X = datos[["Largo_Petalo", "Ancho_Petalo"]]
y = datos["Tipo_Flor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

precision = precision_score(y_test, y_pred, average='macro')
sensibilidad = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("=== Resultados del modelo KNN ===")
print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precisión (Precision): {precision:.2f}")
print(f"Sensibilidad (Recall): {sensibilidad:.2f}")
print(f"F1-Score: {f1:.2f}")

patrones = pd.DataFrame({
    "Largo_Petalo": [5.1235, 4.9842, 6.1213, 2.3123],
    "Ancho_Petalo": [6.8832, 7.8921, 4.1235, 8.1231]
})

predicciones = knn.predict(patrones)

print("\n=== Predicciones para patrones de entrada ===")
for i, p in enumerate(predicciones):
    print(f"P{i+1} → Tipo_Flor: {p}")

plt.figure(figsize=(7,5))
plt.title("Diagrama de dispersión - Clasificación de Flores (KNN)")
plt.scatter(X["Largo_Petalo"], X["Ancho_Petalo"], c=y, cmap="viridis", label="Datos originales", alpha=0.7)
plt.scatter(patrones["Largo_Petalo"], patrones["Ancho_Petalo"], c='red', marker='x', s=100, label="Nuevos patrones")
plt.xlabel("Largo_Petalo")
plt.ylabel("Ancho_Petalo")
plt.legend()
plt.grid(True)
plt.show()
