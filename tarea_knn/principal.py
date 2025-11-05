import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter

# --- Clase Flor ---
class Flor:
    def __init__(self, largo, ancho, tipo):
        self.largo = largo
        self.ancho = ancho
        self.tipo = tipo


# --- Leer archivo Excel ---
def leer_excel(ruta):
    df = pd.read_excel(ruta)
    flores = []
    for _, row in df.iterrows():
        flores.append(Flor(row[0], row[1], row[2]))
    return flores


# --- KNN básico ---
def clasificar_knn(datos, largo, ancho, k):
    distancias = []
    for f in datos:
        distancia = math.sqrt((f.largo - largo) ** 2 + (f.ancho - ancho) ** 2)
        distancias.append((distancia, f.tipo))

    # Ordenar por distancia
    distancias.sort(key=lambda x: x[0])

    # Tomar los k vecinos más cercanos
    vecinos = [tipo for _, tipo in distancias[:k]]

    # Contar la clase más común
    tipo_predicho = Counter(vecinos).most_common(1)[0][0]
    return tipo_predicho


# --- Mostrar gráfica ---
def mostrar_grafica(flores):
    colores = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}

    for f in flores:
        plt.scatter(f.largo, f.ancho, color=colores.get(f.tipo, 'black'), label=f.tipo)

    plt.xlabel("Largo del pétalo")
    plt.ylabel("Ancho del pétalo")
    plt.title("Diagrama de dispersión - Flores")
    plt.grid(True)

    # Evitar etiquetas duplicadas en la leyenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


# --- Programa principal ---
def main():
    try:
        archivo = "dataset_flores_ruido.xlsx"  # Usa el nombre real del archivo
        flores = leer_excel(archivo)
        print(f"Se leyeron {len(flores)} flores del archivo.\n")

        # Patrones nuevos
        patrones = [
            [5.1235, 6.8832],
            [4.9842, 7.8921],
            [6.1213, 4.1235],
            [2.3123, 8.1231]
        ]

        k = 3

        for i, p in enumerate(patrones, start=1):
            tipo = clasificar_knn(flores, p[0], p[1], k)
            print(f"Patrón P{i} -> Largo: {p[0]}, Ancho: {p[1]} => Tipo predicho: {tipo}")

        # Métricas de ejemplo
        precision = 0.85
        sensibilidad = 0.83
        f1 = 2 * ((precision * sensibilidad) / (precision + sensibilidad))

        print("\nMétricas (solo ejemplo, no reales del dataset):")
        print(f"Precisión: {precision:.2f}")
        print(f"Sensibilidad: {sensibilidad:.2f}")
        print(f"F1-Score: {f1:.2f}")

        # Mostrar gráfica
        mostrar_grafica(flores)

    except Exception as e:
        print("Ocurrió un error:", e)


if __name__ == "__main__":
    main()
