import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# --- 1. CONFIGURACIÓN Y PREPARACIÓN DE DATOS ---

# 1.1 Definir parámetros (AJUSTA LA RUTA SI ES NECESARIO)
url_file = './data/dataset_flores_ruido.xlsx'
k_vecinos = 3

# 1.2 Carga de Datos
try:
    df = pd.read_excel(url_file)
except FileNotFoundError:
    print(f"Error: El archivo '{url_file}' no fue encontrado. Verifica la ruta.")
    exit()

# 1.3 Seleccionar Características (X) y Etiqueta (y)
X = df[['Largo_Petalo', 'Ancho_Petalo']]
y = df['Tipo_Flor']

# 1.4 Preprocesamiento: Escalar los datos (Crucial para k-NN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1.5 Dividir para Evaluación (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("Datos cargados, seleccionados y escalados correctamente.")
print("-" * 50)


# --- 2. APLICACIÓN DE k-NN Y CLASIFICACIÓN DE PATRONES DE ENTRADA ---

# 2.1 Entrenar el modelo
knn = KNeighborsClassifier(n_neighbors=k_vecinos)
knn.fit(X_train, y_train)

# 2.2 Patrones de entrada (P1, P2, P3, P4)
patrones_entrada = np.array([
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231]
])

# 2.3 Escalar los nuevos patrones
patrones_escalados = scaler.transform(patrones_entrada)

# 2.4 Realizar las predicciones
predicciones = knn.predict(patrones_escalados)

print("## 🧠 Clasificación de Patrones de Entrada")
for i, (patron, prediccion) in enumerate(zip(patrones_entrada, predicciones)):
    print(f"P{i+1}: Largo={patron[0]:.4f}, Ancho={patron[1]:.4f} -> Clasificado como: {prediccion}")
print("-" * 50)


# --- 3. DIAGRAMA DE DISPERSIÓN (SCATTER PLOT) ---

plt.figure(figsize=(10, 6))

# Trazar datos originales (Usando X sin escalar)
scatter = plt.scatter(X['Largo_Petalo'], X['Ancho_Petalo'], 
                      c=pd.factorize(y)[0], cmap='viridis', s=50)

# Trazar los 4 nuevos patrones (Usando patrones_entrada sin escalar)
plt.scatter(patrones_entrada[:, 0], patrones_entrada[:, 1], 
            marker='X', s=200, color='red', label='Nuevos Patrones (P1-P4)')

# Etiquetas y Título
plt.xlabel('Largo_Petalo')
plt.ylabel('Ancho_Petalo')
plt.title(f'Diagrama de Dispersión de Flores (k-NN con K={k_vecinos})')

# Leyendas
legend1 = plt.legend(*scatter.legend_elements(), title="Tipo_Flor", loc="lower right")
plt.gca().add_artist(legend1)
plt.legend(loc="upper left")

plt.grid(True)
plt.show() 


# --- 4. SALIDA ESTÁNDAR Y MÉTRICAS DE EVALUACIÓN ---

# 4.1 Predicción sobre el conjunto de prueba
y_pred = knn.predict(X_test)

# 4.2 Reporte de Clasificación (incluye las métricas requeridas)
reporte = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

print("##Métricas de Evaluación ")
print(f"{'Clase':<15} {'Precisión':>10} {'Sensibilidad':>15} {'F1-score':>10}")
print("-" * 55)

# 4.3 Mostrar métricas por clase  