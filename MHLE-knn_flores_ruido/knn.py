import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score

data = pd.read_excel("dataset_flores_ruido.xlsx")

print("=== Datos cargados ===")
print(data.head())

X = data[['Largo_Petalo', 'Ancho_Petalo']]
y = data['Tipo_Flor']

# Codificar etiquetas (de texto a números)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Crear y entrenar el modelo KNN
k = 3  # número de vecinos
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\n=== Métricas de evaluación ===")
print("Accuracy (Precisión General):", round(accuracy_score(y_test, y_pred), 3))
print("Recall (Sensibilidad):", round(recall_score(y_test, y_pred, average='macro'), 3))
print("F1-Score (Balance):", round(f1_score(y_test, y_pred, average='macro'), 3))
print("Presicion:" , round(precision_score(y_test, y_pred, average='macro'), 3))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

print("Notas:")
print("Accuracy = rendimiento general.")
print("Macro avg = promedio equilibrado entre clases.")
print("Weighted avg = promedio ajustado según el tamaño de cada clase.")

# Clasificando los patrones dados
patrones = pd.DataFrame([
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231]
], columns=['Largo_Petalo', 'Ancho_Petalo'])

predicciones = knn.predict(patrones)
nombres_pred = encoder.inverse_transform(predicciones)

print("\n=== Clasificación de nuevos patrones ===")
for i, nombre in enumerate(nombres_pred):
    print(f"P{i+1}: {nombre}")

# Graficando el diagrama de dispersión
plt.figure(figsize=(8,6))
for tipo in np.unique(y_encoded):
    plt.scatter(
        X.loc[y_encoded == tipo, 'Largo_Petalo'],
        X.loc[y_encoded == tipo, 'Ancho_Petalo'],
        label=encoder.inverse_transform([tipo])[0]
    )

# Agregar los nuevos patrones
plt.scatter(patrones['Largo_Petalo'], patrones['Ancho_Petalo'],
            color='black', marker='x', s=100, label='Patrones nuevos')

plt.title("Diagrama de dispersión - Clasificación KNN")
plt.xlabel("Largo del pétalo")
plt.ylabel("Ancho del pétalo")
plt.legend()
plt.grid(True)
plt.show()
