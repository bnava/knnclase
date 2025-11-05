# 🌸 Clasificador de Flores con KNN (versión Brandon)
# Requiere: pandas, scikit-learn, matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# === 1. Cargar datos ===
archivo = "C:/Users/Laptop/OneDrive/Desktop/launch.json/Myflowers.xlsx"

try:
    datos = pd.read_excel(archivo)
    print(f"✅ Datos cargados desde: {archivo}")
except FileNotFoundError:
    print(f"⚠️ No se encontró el archivo '{archivo}'")
    exit()

# === 2. Variables ===
entradas = datos[["Largo_Petalo", "Ancho_Petalo"]]
salida = datos["Tipo_Flor"]

print("\nVista previa de los datos:")
print(datos.sample(5))
print("-" * 50)

# === 3. Normalizar los datos ===
# Aquí uso MinMaxScaler (diferente al StandardScaler)
normalizador = MinMaxScaler()
X_norm = normalizador.fit_transform(entradas)

# === 4. División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, salida, test_size=0.25, random_state=0
)

print(f"Datos divididos: {len(X_train)} entrenamiento / {len(X_test)} prueba")
print("-" * 50)

# === 5. Entrenar el modelo KNN ===
k = 4  # puedes cambiarlo a 3, 5, etc.
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_train, y_train)

print(f"🤖 Modelo KNN entrenado con k={k}")

# === 6. Evaluar el modelo ===
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)

print(f"Precisión del modelo: {precision*100:.2f}%")
print("Matriz de confusión:\n", matriz)
print("-" * 50)

# === 7. Clasificar nuevos ejemplos ===
nuevos = pd.DataFrame({
    "Largo_Petalo": [5.2, 3.1, 6.0],
    "Ancho_Petalo": [2.0, 1.2, 2.4]
})

nuevos_norm = normalizador.transform(nuevos)
pred_nuevos = modelo.predict(nuevos_norm)

print("🌼 Clasificación de nuevos datos:")
for i, clase in enumerate(pred_nuevos, start=1):
    print(f"Flor {i}: {nuevos.iloc[i-1].to_dict()} → {clase}")

# === 8. Visualización simple ===
plt.figure(figsize=(8, 6))
colores = pd.factorize(salida)[0]

plt.scatter(
    entradas["Largo_Petalo"], entradas["Ancho_Petalo"],
    c=colores, cmap="coolwarm", s=60, edgecolor='k', alpha=0.7
)

plt.scatter(
    nuevos["Largo_Petalo"], nuevos["Ancho_Petalo"],
    marker="*", s=200, color="gold", label="Nuevos datos"
)

plt.title(f"Clasificación de flores con KNN (k={k})")
plt.xlabel("Largo del pétalo")
plt.ylabel("Ancho del pétalo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
