import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

print("✅ El archivo se está ejecutando correctamente")
print("Leyendo archivo Excel...")
data = pd.read_excel("dataset_flores_ruido.xlsx")

print("\nDatos cargados correctamente:")
print(data.head())

columnas_necesarias = ["Largo_Petalo", "Ancho_Petalo", "Tipo_Flor"]
for col in columnas_necesarias:
    if col not in data.columns:
        raise ValueError(f"Falta la columna: {col}")


X = data[["Largo_Petalo", "Ancho_Petalo"]]
y = data["Tipo_Flor"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


print("\nEntrenando modelo KNN (k=3)...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Modelo entrenado correctamente.")

patrones = [
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231]
]

predicciones = knn.predict(patrones)

print("\nPredicciones de los patrones:")
for i, (p, pred) in enumerate(zip(patrones, predicciones), start=1):
    print(f"P{i}: {p} → Tipo de flor predicho: {pred}")

print("\nCalculando métricas del modelo...")
y_pred = knn.predict(X_test)
precision = accuracy_score(y_test, y_pred)
sensibilidad = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n--- RESULTADOS ---")
print(f"Precisión: {precision:.2f}")
print(f"Sensibilidad: {sensibilidad:.2f}")
print(f"F1-score: {f1:.2f}")

print("\n🖼️ Mostrando diagrama de dispersión...")

tipos = data["Tipo_Flor"].unique()
colores = plt.cm.rainbow(range(len(tipos)))

plt.figure(figsize=(8,6))
for tipo, color in zip(tipos, colores):
    subset = data[data["Tipo_Flor"] == tipo]
    plt.scatter(subset["Largo_Petalo"], subset["Ancho_Petalo"], label=tipo, color=color)

plt.title("Clasificación de Flores (KNN)")
plt.xlabel("Largo del Pétalo")
plt.ylabel("Ancho del Pétalo")
plt.legend()
plt.grid(True)
plt.show()
