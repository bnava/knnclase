############################################################################################
#En el documento Excel se integró una hoja nueva llamada "NuevosPatrones"                  #
# donde se agregaron los patrones de entrada proporcionados en el PDF de la tarea          #
# (P1–P4 con sus valores de Largo_Petalo y Ancho_Petalo), la primera columna (columna 0)   #
# del documento contiene los identificadores de cada punto a clasificar (P1, P2, P3 y P4), #
# y se utiliza como indice del DataFrame al leer esta hoja.                                #
############################################################################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df = pd.read_excel("C:/Users/santo/OneDrive/Documentos/dataset_flores_ruido-1.xlsx", sheet_name= "Sheet1")
df_new = pd.read_excel("C:/Users/santo/OneDrive/Documentos/dataset_flores_ruido-1.xlsx", sheet_name="NuevosPatrones", index_col=0)

print("\nDatos cargados:")
print(df.head())

print("\nDatos a clasificar:")
print(df_new)

print("\nColumnas")
X = df[["Largo_Petalo", "Ancho_Petalo"]]
y = df["Tipo_Flor"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)
print("Reporte clasificacion:")
print(classification_report(y_test, y_pred))


pred_new = knn.predict(df_new)


print("\nPredicciones para nuevos patrones:")
for idx, pred in zip(df_new.index, pred_new):
    print(f"{idx} -> {pred}")


classes = y.unique()
colors =cm.viridis(np.linspace(0, 1, len(classes)))

plt.figure(figsize=(8,6))

for cls, color in zip(classes, colors):
    subset = X[y == cls]
    plt.scatter(subset["Largo_Petalo"], subset["Ancho_Petalo"],
                label=f"Clase {cls}", color = color, s=80, alpha=0.7)


plt.scatter(df_new["Largo_Petalo"], df_new["Ancho_Petalo"], c='red', marker='X', s=120, label="Nuevos patrones")

for idx, row in df_new.iterrows():
    plt.text(row["Largo_Petalo"] + 0.05, row["Ancho_Petalo"] + 0.05, idx,
             color='black', fontsize=10, weight='bold')

plt.xlabel("Largo del pétalo")
plt.ylabel("Ancho del pétalo")
plt.title("Clasificacion KNN con nuevos patrones desde Excel")
plt.legend()
plt.grid(True)
plt.show()


