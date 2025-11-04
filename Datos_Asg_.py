"""by asg: Ocupamos-> pandas, numpy, scikit-learn, matplotlib, openpyxl"""

# librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# LECTURA

ruta_archivo = "./dataset_flores_ruido.xlsx" #path
num_vecinos = 5  #k

try:
    datos = pd.read_excel(ruta_archivo)
    print(f"Archivo leído correctamente: {ruta_archivo}")
except FileNotFoundError:
    print(f"⚠️ Error: no se encontró el archivo '{ruta_archivo}'. Verifica la ruta o el nombre.")
    exit()

# Seleccionar variables
X = datos[["Largo_Petalo", "Ancho_Petalo"]]
y = datos["Tipo_Flor"]

print("\nPrimeras filas del conjunto de datos:")
print(datos.head())
print("-" * 60)

# normalizacion
escalador = StandardScaler()
X_norm = escalador.fit_transform(X)

# División de los datos 
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42, stratify=y
)
print("Datos divididos en entrenamiento y prueba correctamente.")
print("-" * 60)


# ENTRENAMIENTO

# Crear el modelo 
modelo_knn = KNeighborsClassifier(n_neighbors=num_vecinos)
modelo_knn.fit(X_train, y_train)
print(f"Modelo KNN entrenado exitosamente con k = {num_vecinos}.")
print("-" * 60)

# Patrones de entrada nuevos
nuevos_patrones = np.array([
    [5.1235, 6.8832],  
    [4.9842, 7.8921],  
    [6.1213, 4.1235],  
    [2.3123, 8.1231],  
])

#  nuevos datos
nuevos_patrones_scaled = escalador.transform(nuevos_patrones)

#  nuevos patrones
predicciones_nuevos = modelo_knn.predict(nuevos_patrones_scaled)

print(" Clasificación de los nuevos patrones:")
for i, (punto, clase_pred) in enumerate(zip(nuevos_patrones, predicciones_nuevos), start=1):
    print(f"P{i}: Largo={punto[0]:.4f}, Ancho={punto[1]:.4f} → Clase predicha: {clase_pred}")
print("-" * 60)




plt.figure(figsize=(10, 6))

# Grafica
colores = pd.factorize(y)[0]
dispersion = plt.scatter(
    X["Largo_Petalo"], X["Ancho_Petalo"], c=colores, cmap="viridis", s=50, edgecolor='k'
)

# patrones de prueba
plt.scatter(
    nuevos_patrones[:, 0], nuevos_patrones[:, 1],
    marker="*", s=200, color="red", label="Patrones P1–P4"
)

plt.title(f"Clasificación KNN de Flores (k = {num_vecinos})")
plt.xlabel("Largo del Pétalo")
plt.ylabel("Ancho del Pétalo")

# cuadrícula
leyenda = plt.legend(*dispersion.legend_elements(), title="Tipo_Flor", loc="lower right")
plt.gca().add_artist(leyenda)
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


#MODELO 

# Predicciones 
y_pred = modelo_knn.predict(X_test)


reporte = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

print("\n MÉTRICAS DE EVALUACIN DEL MODELO")
print(f"{'Clase':<15}{'Precisión':>12}{'Sensibilidad':>15}{'F1-score':>12}")
print("-" * 60)

for etiqueta, valores in reporte.items():
    if etiqueta in ["accuracy", "macro avg", "weighted avg"]:
        continue
    print(f"{etiqueta:<15}{valores['precision']*100:12.2f}{valores['recall']*100:15.2f}{valores['f1-score']*100:12.2f}")

print("-" * 60)
print(f"Precisión global: {reporte['accuracy']*100:.2f}%")
print("Ejecución finalizada correctamente ")
