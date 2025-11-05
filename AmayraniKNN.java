import org.apache.poi.ss.usermodel.*;
import java.io.*;
import java.util.*;

public class Flor {
    double x;
    double y;
    String tipo;

    public Flor(double x, double y, String tipo) {
        this.x = x;
        this.y = y;
        this.tipo = tipo;
    }
}

public class LeerExcel {
    public static List<Flor> leerDatos(String ruta) {
        List<Flor> datos = new ArrayList<>();

        try (FileInputStream file = new FileInputStream(new File(ruta));
             Workbook workbook = WorkbookFactory.create(file)) {

            Sheet hoja = workbook.getSheetAt(0);
            for (Row fila : hoja) {
                if (fila.getRowNum() == 0) continue; // Saltar encabezado

                double x = fila.getCell(0).getNumericCellValue();
                double y = fila.getCell(1).getNumericCellValue();
                String tipo = fila.getCell(2).getStringCellValue();

                datos.add(new Flor(x, y, tipo));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return datos;
    }
}

public class KNN {
    public static String clasificar(List<Flor> datos, Flor nueva, int k) {
        // Calcular distancia euclidiana
        List<FlorDistancia> distancias = new ArrayList<>();
        for (Flor f : datos) {
            double d = Math.sqrt(Math.pow(f.x - nueva.x, 2) + Math.pow(f.y - nueva.y, 2));
            distancias.add(new FlorDistancia(f, d));
        }

        // Ordenar por distancia
        distancias.sort((a, b) -> Double.compare(a.distancia, b.distancia));

        // Tomar los k vecinos más cercanos
        Map<String, Integer> conteo = new HashMap<>();
        for (int i = 0; i < k; i++) {
            String tipo = distancias.get(i).flor.tipo;
            conteo.put(tipo, conteo.getOrDefault(tipo, 0) + 1);
        }

        // Devolver la clase más frecuente
        return Collections.max(conteo.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // Clase auxiliar
    static class FlorDistancia {
        Flor flor;
        double distancia;

        FlorDistancia(Flor flor, double distancia) {
            this.flor = flor;
            this.distancia = distancia;
        }
    }
}

public class Evaluacion {
    public static double calcularAccuracy(List<Flor> datos, int k) {
        int aciertos = 0;

        for (int i = 0; i < datos.size(); i++) {
            List<Flor> copia = new ArrayList<>(datos);
            Flor actual = copia.remove(i); // quitamos la flor actual
            String prediccion = KNN.clasificar(copia, actual, k);

            if (prediccion.equals(actual.tipo)) {
                aciertos++;
            }
        }
        return (double) aciertos / datos.size();
    }
}



public class Practica {
    public static void main(String[] args) {
        List<Flor> datos = LeerExcel.leerDatos("\"C:\\Users\\amayr\\OneDrive\\Escritorio\\dataset_flores_ruido.xlsx\"");

        // Clasificar una nueva flor
        Flor nueva = new Flor(5.0, 3.0, null);
        String tipo = KNN.clasificar(datos, nueva, 3);
        System.out.println("La nueva flor fue clasificada como: " + tipo);

        // Calcular exactitud del modelo
        double accuracy = Evaluacion.calcularAccuracy(datos, 3);
        System.out.printf("Exactitud del modelo: %.2f%%\n", accuracy * 100);
    }
}

