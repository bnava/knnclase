import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

direccionExel = 'dataset_flores_ruido.xlsx'
vecinosK = 3

try:
    df = pd.read_excel(direccionExel)
except FileNotFoundError:
    print(f"Error: '{direccionExel}' no encontrado.")
    exit()

X = df[['Largo_Petalo', 'Ancho_Petalo']]
y = df['Tipo_Flor']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"Datos son correctos")

knn = KNeighborsClassifier(n_neighbors=vecinosK)
knn.fit(X_train, y_train)

patronesEntrantes = np.array([
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231]
])

patrones_df = pd.DataFrame(patronesEntrantes, columns=X.columns)
patronesEscalados = scaler.transform(patrones_df)

predicciones = knn.predict(patronesEscalados)

print(f"Clasificación de Patrones de Entrada")

for i, (patron, prediccion) in enumerate(zip(patronesEntrantes, predicciones)):
    print(f"P{i+1}: Largo = {patron[0]:.4f}, Ancho= {patron[1]:.4f} --- Clasificado como: {prediccion}")
print()


plt.figure(figsize=(10, 6))

scatter = plt.scatter(X['Largo_Petalo'], X['Ancho_Petalo'],
                      c=pd.factorize(y)[0], cmap='viridis', s=50)

plt.scatter(patronesEntrantes[:, 0], patronesEntrantes[:, 1],
            marker='X', s=200, color='red', label='Nuevos Patrones (P1-P4)')

plt.xlabel('Largo')
plt.ylabel('Ancho')
plt.title(f'Diagrama de Dispersión')

legend1 = plt.legend(*scatter.legend_elements(), title="Tipo_Flor", loc="lower right")
plt.gca().add_artist(legend1)
plt.legend(loc="upper left")

plt.grid(True)
plt.show()

y_pred = knn.predict(X_test)

reporte = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
