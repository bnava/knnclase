#  Clasificación de Flores con KNN

##  Descripción
Este programa aplica el algoritmo **K-Nearest Neighbors (KNN)** para clasificar flores según las características de sus pétalos.  
Usa un archivo Excel (`flores.xlsx`) que contiene las columnas:
- `Largo_Petalo`
- `Ancho_Petalo`
- `Tipo_Flor`

El programa entrena el modelo con esos datos y luego clasifica cuatro nuevos patrones de entrada.

---

## Algoritmo
Se usa el algoritmo **KNN (k = 3)**, que clasifica un punto nuevo según las clases más frecuentes entre sus vecinos más cercanos.

Los patrones a clasificar son:

| Patrón | Largo_Petalo | Ancho_Petalo |
|:--:|:--:|:--:|
| P1 | 5.1235 | 6.8832 |
| P2 | 4.9842 | 7.8921 |
| P3 | 6.1213 | 4.1235 |
| P4 | 2.3123 | 8.1231 |


# Salida esperada
En consola:
```
=== Reporte de Clasificación ===
              precision    recall  f1-score   support
Setosa          1.00       1.00      1.00         3
Versicolor      1.00       0.67      0.80         3
Virginica       0.75       1.00      0.86         3

=== Clasificación de los nuevos patrones ===
P1 -> Virginica
P2 -> Virginica
P3 -> Versicolor
P4 -> Versicolor
```