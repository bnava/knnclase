import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys

# 🎨 Colores automáticos por nombre de línea
LINE_COLORS = {}

def get_color_for_line(linea):
    """Asigna colores a cada línea automáticamente."""
    base_colors = [
        "red", "blue", "green", "orange", "purple",
        "brown", "pink", "gray", "olive", "cyan"
    ]
    if linea not in LINE_COLORS:
        LINE_COLORS[linea] = base_colors[len(LINE_COLORS) % len(base_colors)]
    return LINE_COLORS[linea]


def load_stations_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Archivo '{file_path}' cargado correctamente.")
        print(f"Columnas detectadas: {list(df.columns)}\n")
        return df
    except Exception as e:
        print(f"❌ Error al leer el archivo Excel: {e}")
        sys.exit(1)


def build_network_from_lines(df):
    G = nx.Graph()

    for col in df.columns:
        linea = col.strip()
        estaciones = df[col].dropna().tolist()

        for i in range(len(estaciones) - 1):
            est_actual = str(estaciones[i]).strip()
            est_siguiente = str(estaciones[i + 1]).strip()

            if not G.has_node(est_actual):
                G.add_node(est_actual, lines=set())
            if not G.has_node(est_siguiente):
                G.add_node(est_siguiente, lines=set())

            G.nodes[est_actual]['lines'].add(linea)
            G.nodes[est_siguiente]['lines'].add(linea)

            if not G.has_edge(est_actual, est_siguiente):
                G.add_edge(est_actual, est_siguiente, weight=3.0, lines={linea})
            else:
                G[est_actual][est_siguiente]['lines'].add(linea)

    print(f"🚏 Red creada con {len(G.nodes)} estaciones y {len(G.edges)} conexiones.")
    return G



def suggest_route(G, origin, destination, plot=True, solo_ruta=True):

    if origin not in G.nodes:
        print(f"❌ La estación de origen '{origin}' no existe.")
        return
    if destination not in G.nodes:
        print(f"❌ La estación de destino '{destination}' no existe.")
        return

    try:
        path = nx.shortest_path(G, origin, destination, weight='weight')
        total_time = nx.shortest_path_length(G, origin, destination, weight='weight')
    except nx.NetworkXNoPath:
        print("🚫 No hay conexión entre esas estaciones.")
        return

    print("\n✅ Ruta más rápida encontrada:")
    print("  → ".join(path))
    print(f"⏱ Tiempo estimado total: {total_time:.1f} minutos\n")

    if plot:
        plt.figure(figsize=(11, 6))

        route_graph = nx.Graph()
        route_edges = list(zip(path[:-1], path[1:]))

        route_graph.add_nodes_from(path)
        route_graph.add_edges_from(route_edges)

        pos = nx.spring_layout(route_graph, seed=42)

        # Dibujar nodos
        nx.draw_networkx_nodes(route_graph, pos, node_size=700, node_color='lightgray')

        # 🎨 Colorear cada tramo por la línea correspondiente
        for u, v in route_edges:
            lineas = list(G[u][v]['lines'])
            linea = lineas[0]  # tomar la primera línea del tramo
            color = get_color_for_line(linea)

            nx.draw_networkx_edges(
                route_graph,
                pos,
                edgelist=[(u, v)],
                width=3,
                edge_color=color
            )

        nx.draw_networkx_labels(route_graph, pos, font_size=9)

        plt.title(f"Ruta: {origin} → {destination}  ({total_time:.1f} min)")
        plt.axis('off')
        plt.show()



def main():
    file_path = "LDELM.xlsx"
    df = load_stations_from_excel(file_path)
    G = build_network_from_lines(df)

    while True:
        print("Escribe correctamente la estación de origen y de destino")

        origen = input("\nEstación de origen, o (salir / exit): ").strip()
        if origen.lower() in ['salir', 'exit', 'q']:
            print("👋 Saliendo del programa.")
            break

        destino = input("Estación de destino: ").strip()

        suggest_route(G, origen, destino, plot=True, solo_ruta=True)


if __name__ == "__main__":
    main()

