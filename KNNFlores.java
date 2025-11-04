import java.io.*;
import java.util.*;

public class KNNFlores {
    public static void main(String[] args) {
        List<double[]> datos = new ArrayList<>();
        List<String> clases = new ArrayList<>();

        // Lee el archivo CSV en lugar del Excel
        try (BufferedReader br = new BufferedReader(new FileReader("dataset_flores_ruido.csv"))) {
            String linea;
            br.readLine(); // Saltar encabezado
            while ((linea = br.readLine()) != null) {
                String[] partes = linea.split(",");
                if (partes.length < 3) continue;
                double largo = Double.parseDouble(partes[0]);
                double ancho = Double.parseDouble(partes[1]);
                String tipo = partes[2];
                datos.add(new double[]{largo, ancho});
                clases.add(tipo);
            }
        } catch (Exception e) {
            System.out.println(" Error al leer el archivo: " + e.getMessage());
            return;
        }

        if (datos.isEmpty()) {
            System.out.println(" No se encontraron datos en dataset_flores_ruido.csv");
            return;
        }

        Scanner sc = new Scanner(System.in);
        System.out.print("Introduce el largo del pétalo: ");
        double largo = sc.nextDouble();
        System.out.print("Introduce el ancho del pétalo: ");
        double ancho = sc.nextDouble();

        String clasePredicha = knnClasificar(datos, clases, largo, ancho, 3);
        System.out.println("\n La flor clasificada es: " + clasePredicha);
    }

    public static String knnClasificar(List<double[]> datos, List<String> clases,
                                       double largo, double ancho, int k) {

        List<double[]> distancias = new ArrayList<>();

        for (int i = 0; i < datos.size(); i++) {
            double[] punto = datos.get(i);
            double distancia = Math.sqrt(Math.pow(punto[0] - largo, 2) + Math.pow(punto[1] - ancho, 2));
            distancias.add(new double[]{distancia, i});
        }

        distancias.sort(Comparator.comparingDouble(a -> a[0]));

        Map<String, Integer> contador = new HashMap<>();
        for (int i = 0; i < k && i < distancias.size(); i++) {
            int indice = (int) distancias.get(i)[1];
            String clase = clases.get(indice);
            contador.put(clase, contador.getOrDefault(clase, 0) + 1);
        }

        String claseGanadora = "";
        int max = -1;
        for (Map.Entry<String, Integer> entry : contador.entrySet()) {
            if (entry.getValue() > max) {
                max = entry.getValue();
                claseGanadora = entry.getKey();
            }
        }

        return claseGanadora;
    }
}