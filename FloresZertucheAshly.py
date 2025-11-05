import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

ruta_datos = "./dataset_flores_ruido.xlsx"

try:
    df = pd.read_excel(ruta_datos)
    print(f"Archivo leído correctamente: {ruta_datos}")
except FileNotFoundError:
    print(f"⚠️ Error: no se encontró el archivo '{ruta_datos}'. Verifica la ruta o el nombre.")
    exit()

X = df[["Largo_Petalo", "Ancho_Petalo"]]
y = df["Tipo_Flor"]

print("\nPrimeras filas del conjunto de datos:")
print(df.head())
print("-" * 60)

escalador = StandardScaler()
X_normalizado = escalador.fit_transform(X)

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X_normalizado, y, test_size=0.3, random_state=42, stratify=y
)

print("Datos divididos correctamente.")
print("-" * 60)

valor_k = 3
modelo = KNeighborsClassifier(n_neighbors=valor_k)
modelo.fit(X_entrenamiento, y_entrenamiento)
print(f"Modelo KNN entrenado con k = {valor_k}.")
print("-" * 60)

nuevos_datos = np.array([
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231],
])

nuevos_datos_norm = escalador.transform(nuevos_datos)
predicciones = modelo.predict(nuevos_datos_norm)

print("Clasificación de nuevos patrones:")
for i, (punto, pred) in enumerate(zip(nuevos_datos, predicciones), start=1):
    print(f"P{i}: Largo={punto[0]:.4f}, Ancho={punto[1]:.4f} → Clase: {pred}")
print("-" * 60)

plt.figure(figsize=(10, 6))
colores = pd.factorize(y)[0]
dispersión = plt.scatter(
    X["Largo_Petalo"], X["Ancho_Petalo"], c=colores, cmap="viridis", s=50, edgecolor='k'
)

plt.scatter(
    nuevos_datos[:, 0], nuevos_datos[:, 1],
    marker="*", s=200, color="red", label="Nuevos Patrones"
)

plt.title(f"Clasificación KNN (k = {valor_k})")
plt.xlabel("Largo del Pétalo")
plt.ylabel("Ancho del Pétalo")
leyenda = plt.legend(*dispersión.legend_elements(), title="Tipo_Flor", loc="lower right")
plt.gca().add_artist(leyenda)
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

predicciones_prueba = modelo.predict(X_prueba)
reporte = classification_report(y_prueba, predicciones_prueba, zero_division=0, output_dict=True)

print("\nMÉTRICAS DEL MODELO")
print(f"{'Clase':<15}{'Precisión':>12}{'Sensibilidad':>15}{'F1-score':>12}")
print("-" * 60)

for etiqueta, valores in reporte.items():
    if etiqueta in ["accuracy", "macro avg", "weighted avg"]:
        continue
    print(f"{etiqueta:<15}{valores['precision']*100:12.2f}{valores['recall']*100:15.2f}{valores['f1-score']*100:12.2f}")

print("-" * 60)
print(f"Precisión global: {reporte['accuracy']*100:.2f}%")
print("Ejecución finalizada correctamente.")
