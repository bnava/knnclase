import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ===== Funciones =====
def euclidean_distance_squared(p1, p2):
    """Distancia al cuadrado entre dos puntos 2D"""
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def knn_predict(train_X, train_y, test_point, k=3):
    """Predice la clase de un punto usando KNN"""
    distances = []
    for i, point in enumerate(train_X):
        d2 = euclidean_distance_squared(point, test_point)
        distances.append((d2, train_y[i], point))
    
    # Ordenar por distancia
    distances.sort(key=lambda x: x[0])
    
    # Tomar los k vecinos más cercanos
    neighbors = distances[:k]
    
    # Mostrar en consola
    print(f"\nTest point: {test_point}")
    print("Vecinos más cercanos (distancia al cuadrado, clase):")
    for d2, cls, pt in neighbors:
        print(f"{pt} d²={d2} ({cls})")
    
    # Votación
    votes = [cls for _, cls, _ in neighbors]
    pred = Counter(votes).most_common(1)[0][0]
    print(f"Votos: {votes} → Predicción = {pred}")
    return pred

def confusion_matrix_metrics(y_true, y_pred):
    """Calcula matriz de confusión y métricas"""
    TP = sum((yt==1 and yp==1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt==0 and yp==0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt==0 and yp==1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt==1 and yp==0) for yt, yp in zip(y_true, y_pred))
    
    print("\nMatriz de Confusión:")
    print(f"\tPred: 1\tPred: 0")
    print(f"Real: 1\tTP={TP}\tFN={FN}")
    print(f"Real: 0\tFP={FP}\tTN={TN}")
    
    accuracy = (TP+TN)/len(y_true)
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    print(f"\nMétricas:\nAccuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return TP, TN, FP, FN, accuracy, precision, recall, f1

# ===== Ejemplo 1: clusters bien separados =====
print("=== Ejemplo 1: clusters bien separados ===")
train_X1 = [(1,1),(2,1),(1,2),(2,2),(7,7),(8,7),(7,8),(8,8)]
train_y1 = [0,0,0,0,1,1,1,1]
test_X1 = [(1.5,1.5),(7.5,7.5),(3,3),(6,6),(4,4),(5.5,5.5)]
test_y1 = [0,1,0,1,0,1]

predictions1 = []
for pt in test_X1:
    pred = knn_predict(train_X1, train_y1, pt, k=3)
    predictions1.append(pred)

confusion_matrix_metrics(test_y1, predictions1)

# Graficar
plt.figure(figsize=(8,6))
train_X1_np = np.array(train_X1)
test_X1_np = np.array(test_X1)
predictions1_np = np.array(predictions1)

# Puntos de entrenamiento
plt.scatter(train_X1_np[:4,0], train_X1_np[:4,1], c='blue', label='Clase 0 (train)')
plt.scatter(train_X1_np[4:,0], train_X1_np[4:,1], c='red', label='Clase 1 (train)')

# Puntos de prueba
for i, pt in enumerate(test_X1_np):
    color = 'blue' if predictions1[i]==0 else 'red'
    marker = 'o' if test_y1[i]==predictions1[i] else 'x'
    plt.scatter(pt[0], pt[1], c=color, marker=marker, s=100, edgecolor='black')

plt.title("Ejemplo 1: clusters bien separados (K=3)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()


# ===== Ejemplo 2: solapamiento / ruido =====
print("\n=== Ejemplo 2: solapamiento / ruido ===")
train_X2 = [(1,1),(2,2),(1.5,1.5),(6,6),(7,7),(8,8)]
train_y2 = [0,0,1,0,1,1]
test_X2 = [(1.4,1.6),(5.2,5.2),(7.2,7.1),(2.5,2.5)]
test_y2 = [0,0,1,0]

predictions2 = []
for pt in test_X2:
    pred = knn_predict(train_X2, train_y2, pt, k=3)
    predictions2.append(pred)

confusion_matrix_metrics(test_y2, predictions2)

# Graficar
plt.figure(figsize=(8,6))
train_X2_np = np.array(train_X2)
test_X2_np = np.array(test_X2)

# Puntos de entrenamiento
plt.scatter(train_X2_np[:2,0], train_X2_np[:2,1], c='blue', label='Clase 0 (train)')
plt.scatter([train_X2_np[2,0]], [train_X2_np[2,1]], c='red', label='Clase 1 ruidoso')
plt.scatter(train_X2_np[3:,0], train_X2_np[3:,1], c='red', label='Clase 1 (train)')

# Puntos de prueba
for i, pt in enumerate(test_X2_np):
    color = 'blue' if predictions2[i]==0 else 'red'
    marker = 'o' if test_y2[i]==predictions2[i] else 'x'
    plt.scatter(pt[0], pt[1], c=color, marker=marker, s=100, edgecolor='black')

plt.title("Ejemplo 2: solapamiento / ruido (K=3)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()