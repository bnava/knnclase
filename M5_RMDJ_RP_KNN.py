from colorama import Fore, Style, init
init(autoreset=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

rutaArchivo = 'dataset_flores_ruido.xlsx'
k_vecinos = 3

try:
    df = pd.read_excel(rutaArchivo)
except FileNotFoundError:
    print(f"{Fore.RED}Tssss Errrror: '{rutaArchivo}' no fue encontrado.")
    exit()

X = df[['Largo_Petalo', 'Ancho_Petalo']]
y = df['Tipo_Flor']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\n{Fore.GREEN}Felicidades, los datos son correctos :D\n")

knn = KNeighborsClassifier(n_neighbors=k_vecinos)
knn.fit(X_train, y_train)

patrones_entrada = np.array([
    [5.1235, 6.8832],
    [4.9842, 7.8921],
    [6.1213, 4.1235],
    [2.3123, 8.1231]
])

patrones_df = pd.DataFrame(patrones_entrada, columns=X.columns)
patrones_escalados = scaler.transform(patrones_df)

predicciones = knn.predict(patrones_escalados)

print(f"{Fore.BLUE}Clasificación de Patrones de Entrada")
print(f"{Fore.CYAN}-" * 75)
for i, (patron, prediccion) in enumerate(zip(patrones_entrada, predicciones)):
    print(f"{Fore.YELLOW}P{i+1}: {Fore.MAGENTA}Largo = {patron[0]:.4f},\tAncho={patron[1]:.4f}\t=>\t\tClasificado como: {prediccion}")
print()


plt.figure(figsize=(10, 6))

scatter = plt.scatter(X['Largo_Petalo'], X['Ancho_Petalo'],
                      c=pd.factorize(y)[0], cmap='viridis', s=50)

plt.scatter(patrones_entrada[:, 0], patrones_entrada[:, 1],
            marker='X', s=200, color='red', label='Nuevos Patrones (P1-P4)')

plt.xlabel('Largo_Petalo')
plt.ylabel('Ancho_Petalo')
plt.title(f'Diagrama de Dispersión de Flores (k-NN con K={k_vecinos})')

legend1 = plt.legend(*scatter.legend_elements(), title="Tipo_Flor", loc="lower right")
plt.gca().add_artist(legend1)
plt.legend(loc="upper left")

plt.grid(True)
plt.show()

y_pred = knn.predict(X_test)

reporte = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

print(f"{Fore.BLUE}Métricas de Evaluación ")
print(f"{Fore.YELLOW}{'Clase':<15} {'Precisión':>10} {'Sensibilidad':>15} {'F1-score':>10}")
print(f"{Fore.CYAN}-" * 75)

for clase, met in reporte.items():
    if clase not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{Fore.MAGENTA}{clase:<15} {met['precision']:>10.2f} {met['recall']:>15.2f} {met['f1-score']:>10.2f}")

print(f"{Fore.CYAN}-" * 75)
print(f"{Fore.YELLOW}Precisión global: {reporte['accuracy']:.2f}")
