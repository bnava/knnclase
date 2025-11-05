# 🌼 Clasificación de Flores usando KNN (versión alternativa)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# === 1. Cargar los datos ===
ruta_excel = "flores_ruido.xlsx"

try:
    df = pd.read_excel(ruta_excel)
    print(f"✅ Archivo cargado correctamente: {ruta_excel}")
except FileNotFoundError:
    print(f"❌ No se encontró el archivo '{ruta_excel}'")
    exit()

print("\nEjemplo de datos cargados:")
print(df.head())
print("-" * 50)

# === 2. Separar variables de entrada y salida ===
X = df[["Largo_Petalo", "Ancho_Petalo"]]
y = df["Tipo_Flor"]

# === 3. Normalizar los valores ===
escalador = MinMaxScaler()
X_escalado = escalador.fit_transform(X)

# === 4. Dividir datos para entrenamiento y prueba ===
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(
    X_escalado, y, test_size=0.25, random_state=42
)

print(f"Datos divididos → Entrenamiento: {len(X_entreno)}, Prueba: {len(X_prueba)}")
print("-" * 50)

# === 5. Crear y entrenar el modelo KNN ===
num_vecinos = 5
knn = KNeighborsClassifier(n_neighbors=num_vecinos)
knn.fit(X_entreno, y_entreno)

print(f"🤖 Modelo KNN entrenado con k = {num_vecinos}")

# === 6. Evaluar el modelo ===
predicciones = knn.predict(X_prueba)
precision = accuracy_score(y_prueba, predicciones)
matriz_conf = confusion_matrix(y_prueba, predicciones)

print(f"🎯 Precisión del modelo: {precision*100:.2f}%")
print("Matriz de confusión:")
print(matriz_conf)
print("-" * 50)

# === 7. Probar con nuevos valores ===
nuevas_muestras = pd.DataFrame({
    "Largo_Petalo": [5.4, 2.9, 6.3],
    "Ancho_Petalo": [2.1, 1.0, 2.5]
})

nuevas_norm = escalador.transform(nuevas_muestras)
pred_nuevas = knn.predict(nuevas_norm)

print("🌸 Clasificación de nuevas flores:")
for idx, tipo in enumerate(pred_nuevas, start=1):
    datos_flor = nuevas_muestras.iloc[idx - 1].to_dict()
    print(f"Flor {idx}: {datos_flor} → Predicción: {tipo}")

# === 8. Visualización gráfica ===
plt.figure(figsize=(8, 6))
colores, etiquetas = pd.factorize(y)

plt.scatter(
    X["Largo_Petalo"], X["Ancho_Petalo"],
    c=colores, cmap="plasma", s=65, edgecolor="black", alpha=0.7, label="Datos originales"
)

plt.scatter(
    nuevas_muestras["Largo_Petalo"], nuevas_muestras["Ancho_Petalo"],
    marker="*", s=220, color="lime", label="Nuevas muestras"
)

plt.title(f"🌺 Clasificación con KNN (k = {num_vecinos})")
plt.xlabel("Largo del pétalo")
plt.ylabel("Ancho del pétalo")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
