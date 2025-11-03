# Cesar Antonio Guillermo Cruz
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Otra opción (si el Excel está en la misma carpeta que el script, en este caso yo puse la ruta completa)
# archivo = "dataset_flores_ruido.xlsx"
archivo = r"C:\Users\cesar\knnclase\dataset_flores_ruido.xlsx"

patrones = np.array([
    [5.1235, 6.8832],  # P1
    [4.9842, 7.8921],  # P2
    [6.1213, 4.1235],  # P3
    [2.3123, 8.1231]   # P4
])

try:
    print(f"Intentando leer: {archivo}")
    datos = pd.read_excel(archivo)
except FileNotFoundError:
    raise SystemExit(f"error: no se encontró el archivo '{archivo}'. Asegúrate de poner la ruta correcta o de copiar el archivo en la misma carpeta.")
except Exception as e:
    raise SystemExit(f"error leyendo el archivo Excel: {e}")

cols_requeridas = ["Largo_Petalo", "Ancho_Petalo", "Tipo_Flor"]
for c in cols_requeridas:
    if c not in datos.columns:
        raise SystemExit(f"error: falta la columna '{c}' en el archivo. Columnas encontradas: {list(datos.columns)}")

datos = datos[cols_requeridas].dropna()
print("\nResumen de los datos cargados")
print(f"Número de registros: {len(datos)}")
print("Tipos de flores y su cantidad:")
print(datos["Tipo_Flor"].value_counts(), "\n")

X = datos[["Largo_Petalo", "Ancho_Petalo"]].values
y_raw = datos["Tipo_Flor"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Mapeo etiquetas (texto -> código):")
for cls, code in zip(le.classes_, range(len(le.classes_))):
    print(f"  {code} -> {cls}")
print("")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y if len(np.unique(y))>1 else None)

k = 3
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("Indicadores (sobre el conjunto de prueba)")
print(f"Precisión (promedio):  {precision_macro:.4f}")
print(f"Sensibilidad (promedio): {recall_macro:.4f}")
print(f"Puntuaje F1 (promedio):   {f1_macro:.4f}\n")

print("Reporte por clase:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
print("Matriz de confusión (filas=verdadero, columnas=predicho):")
print(confusion_matrix(y_test, y_pred))
print("")

print("Clasificación de patrones P1..P4 y detalle de vecinos")
distancias, indices = modelo.kneighbors(patrones, n_neighbors=k, return_distance=True)

for i, (pt, dists, idxs) in enumerate(zip(patrones, distancias, indices)):
    pred = modelo.predict(pt.reshape(1, -1))[0]
    print(f"\nP{i+1} = {pt.tolist()}")
    print(f"  Clase asignada: {le.inverse_transform([pred])[0]} (código {pred})")
    print("  Vecinos más cercanos (desde datos de entrenamiento):")
    for rank, (d, idx) in enumerate(zip(dists, idxs), start=1):
        vecino_feats = X_train[idx]
        vecino_label = le.inverse_transform([y_train[idx]])[0]
        print(f"    {rank}) distancia={d:.4f} -> caracteristicas={vecino_feats.tolist()} , clase={vecino_label}")

print("\nGenerando diagrama de dispersión...")

plt.figure(figsize=(9,7))
unique_codes = np.unique(y)
for code in unique_codes:
    mask = (y == code)
    plt.scatter(X[mask, 0], X[mask, 1], label=le.inverse_transform([code])[0], alpha=0.6)

plt.scatter(patrones[:,0], patrones[:,1], marker='X', s=150, edgecolor='k', linewidth=1.2, label='P1..P4 (patrones)', zorder=5)
for i, p in enumerate(patrones, start=1):
    plt.annotate(f"P{i}", (p[0], p[1]), textcoords="offset points", xytext=(6,-6), fontsize=9, weight='bold')

plt.title(f"Diagrama de dispersión - KNN (k={k})")
plt.xlabel("Largo_Petalo")
plt.ylabel("Ancho_Petalo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nProceso finalizado")
