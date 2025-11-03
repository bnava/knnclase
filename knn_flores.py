import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ===================================
#  Cargando datos del archivo Excel
# ===================================
print("=" * 60)
print("CLASIFICADOR KNN - RECONOCIMIENTO DE PATRONES DE FLORES")
print("=" * 60)

ruta_archivo = "dataset_flores_ruido.xlsx"
print(f"\n[1] Cargando datos desde: {ruta_archivo}")

try:
    df_flores = pd.read_excel(ruta_archivo)
    print(f"    ✓ Archivo cargado exitosamente")
    print(f"    ✓ Total de registros: {len(df_flores)}")
except FileNotFoundError:
    print(f"    ✗ ERROR: No se encontró el archivo '{ruta_archivo}'")
    print(f"    Asegúrate de que el archivo esté en la misma carpeta que este script")
    exit()


print("\n[2] Extrayendo características...")

caracteristicas = df_flores[["Largo_Petalo", "Ancho_Petalo"]].values
etiquetas = df_flores["Tipo_Flor"].values

print(f"    ✓ Características extraídas: Largo_Petalo, Ancho_Petalo")
print(f"    ✓ Clases encontradas: {np.unique(etiquetas)}")

num_vecinos = 3
print(f"\n[3] Entrenando modelo KNN con k={num_vecinos} vecinos...")

clasificador = KNeighborsClassifier(n_neighbors=num_vecinos)
clasificador.fit(caracteristicas, etiquetas)

print(f"    ✓ Modelo entrenado correctamente")


print("\n[4] Clasificando patrones de prueba...")
print("-" * 60)

patrones_prueba = np.array([
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231]
])

resultados_clasificacion = clasificador.predict(patrones_prueba)

print(f"{'Patrón':<10} {'Largo_Petalo':<15} {'Ancho_Petalo':<15} {'Clasificación'}")
print("-" * 60)
for idx, patron in enumerate(patrones_prueba):
    print(f"P{idx+1:<9} {patron[0]:<15.4f} {patron[1]:<15.4f} {resultados_clasificacion[idx]}")


print("\n[5] Evaluando desempeño del modelo...")
print("-" * 60)

predicciones_entrenamiento = clasificador.predict(caracteristicas)
matriz_confusion = confusion_matrix(etiquetas, predicciones_entrenamiento, 
                                   labels=np.unique(etiquetas))

print("\nMatriz de Confusión:")
print(matriz_confusion)

# Calcular métricas para clasificación multiclase
verdaderos_positivos = np.diag(matriz_confusion)
falsos_positivos = matriz_confusion.sum(axis=0) - verdaderos_positivos
falsos_negativos = matriz_confusion.sum(axis=1) - verdaderos_positivos

# Métricas globales (promedio ponderado)
total_muestras = matriz_confusion.sum()
vp_total = verdaderos_positivos.sum()

# Accuracy (Precisión General)
exactitud = vp_total / total_muestras

# Recall y Precision por clase, luego promedio
sensibilidad_por_clase = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
precision_por_clase = verdaderos_positivos / (verdaderos_positivos + falsos_positivos)

# Promedio ponderado según frecuencia de cada clase
pesos_clases = matriz_confusion.sum(axis=1) / total_muestras
sensibilidad_promedio = np.average(sensibilidad_por_clase, weights=pesos_clases)
precision_promedio = np.average(precision_por_clase, weights=pesos_clases)

# F1-Score
puntuacion_f1 = 2 * (precision_promedio * sensibilidad_promedio) / (precision_promedio + sensibilidad_promedio)

print("\n" + "=" * 60)
print("MÉTRICAS DE EVALUACIÓN")
print("=" * 60)
print(f"Accuracy (Precisión General): {exactitud:.4f} ({exactitud*100:.2f}%)")
print(f"Recall (Sensibilidad):        {sensibilidad_promedio:.4f} ({sensibilidad_promedio*100:.2f}%)")
print(f"Precision:                    {precision_promedio:.4f} ({precision_promedio*100:.2f}%)")
print(f"F1-Score (Balance):           {puntuacion_f1:.4f} ({puntuacion_f1*100:.2f}%)")
print("=" * 60)

print("\nMétricas por clase:")
for idx, clase in enumerate(np.unique(etiquetas)):
    f1_clase = 2 * (precision_por_clase[idx] * sensibilidad_por_clase[idx]) / (precision_por_clase[idx] + sensibilidad_por_clase[idx])
    print(f"  {clase}:")
    print(f"    - Sensibilidad: {sensibilidad_por_clase[idx]:.4f}")
    print(f"    - Precisión:    {precision_por_clase[idx]:.4f}")
    print(f"    - F1-Score:     {f1_clase:.4f}")

# ===================================
# PASO 6: Generarando diagrama de dispersión
# ===================================
print("\n[6] Generando diagrama de dispersión...")

plt.figure(figsize=(10, 7))

# Mapeo de colores para las clases
colores_clases = {'Girasol': 'gold', 'Rosa': 'pink', 'Tulipan': 'purple'}
marcadores_clases = {'Girasol': 'o', 'Rosa': 's', 'Tulipan': '^'}

# Graficar datos de entrenamiento
for tipo_flor in np.unique(etiquetas):
    mascara = df_flores["Tipo_Flor"] == tipo_flor
    plt.scatter(
        df_flores[mascara]["Largo_Petalo"],
        df_flores[mascara]["Ancho_Petalo"],
        c=colores_clases.get(tipo_flor, 'gray'),
        marker=marcadores_clases.get(tipo_flor, 'o'),
        label=f'{tipo_flor} (datos entrenamiento)',
        alpha=0.6,
        s=80,
        edgecolors='black',
        linewidth=0.5
    )

# Graficar patrones de prueba
colores_predicciones = [colores_clases.get(pred, 'red') for pred in resultados_clasificacion]
plt.scatter(
    patrones_prueba[:, 0], 
    patrones_prueba[:, 1],
    c=colores_predicciones,
    marker='*',
    s=500,
    label='Patrones de prueba',
    edgecolors='black',
    linewidth=2
)

# Anotar los patrones de prueba
for idx, patron in enumerate(patrones_prueba):
    plt.annotate(
        f'P{idx+1}',
        (patron[0], patron[1]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
    )

plt.title(f'Clasificación de Flores usando KNN (k={num_vecinos})\nAccuracy: {exactitud:.2%}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Largo del Pétalo', fontsize=12)
plt.ylabel('Ancho del Pétalo', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

print("    ✓ Gráfica generada")
print("\n¡Proceso completado exitosamente!")
plt.show()