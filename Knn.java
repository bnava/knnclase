package org.example;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.FileInputStream;
import java.util.*;

public class Knn {
        static class Punto {
            double largo, ancho;
            String tipo;
            String vacio;
            Punto(double l, double a, String t) { largo = l; ancho = a; tipo = t; }
        }


        static double dist(Punto a, Punto b) {
            return Math.sqrt(Math.pow(a.largo - b.largo, 2) + Math.pow(a.ancho - b.ancho, 2));
        }

        // KNN
        static String knn(List<Punto> train, Punto test, int k) {
            List<double[]> dists = new ArrayList<>();
            for (int i = 0; i < train.size(); i++) {
                dists.add(new double[]{dist(train.get(i), test), i});
            }
            dists.sort(Comparator.comparingDouble(arr -> arr[0]));
            HashMap<String, Integer> votos = new HashMap<>();
            for (int i = 0; i < k; i++) {
                String clase = train.get((int) dists.get(i)[1]).tipo;
                votos.put(clase, votos.getOrDefault(clase, 0) + 1);
            }
            return Collections.max(votos.entrySet(), Map.Entry.comparingByValue()).getKey();
        }

        public static void main(String[] args) {
            String nombreArchivo = "C:\\Users\\arett\\OneDrive\\Desktop\\tareaRE\\flores.xlsx";
            List<Punto> datos = new ArrayList<>();

            System.out.println("Leyendo archivo " + nombreArchivo + "...");

            try (FileInputStream fis = new FileInputStream(nombreArchivo);
                 Workbook wb = new XSSFWorkbook(fis)) {

                Sheet hoja = wb.getSheetAt(0);
                Iterator<Row> it = hoja.iterator();
                it.next(); // Saltar encabezado

                while (it.hasNext()) {
                    Row fila = it.next();
                    double largo = fila.getCell(0).getNumericCellValue();
                    double ancho = fila.getCell(1).getNumericCellValue();
                    String tipo = fila.getCell(2).getStringCellValue();
                    datos.add(new Punto(largo, ancho, tipo));
                }

            } catch (Exception e) {
                System.out.println("Error al leer el archivo: " + e.getMessage());
                return;
            }

            System.out.println("Total de patrones leídos: " + datos.size());

            //clases únicas
            HashSet<String> clases = new HashSet<>();
            for (Punto p : datos) clases.add(p.tipo);
            System.out.println("Clases encontradas: " + String.join(", ", clases));

            System.out.println("Dividiendo datos en entrenamiento y prueba...");

            // Mezcla y dividir
            Collections.shuffle(datos, new Random(1));
            int trainSize = (int) (datos.size() * 0.7);
            List<Punto> train = datos.subList(0, trainSize);
            List<Punto> test = datos.subList(trainSize, datos.size());

            int correctos = 0;
            int k = 3;

            System.out.println("Calculando distancias...");
            System.out.println("Evaluando knn con k=" + k + "...");

            for (Punto p : test) {
                if (knn(train, p, k).equals(p.tipo)) correctos++;
            }

            double precision = (double) correctos / test.size();
            double sensibilidad = precision;
            double f1 = (2 * precision * sensibilidad) / (precision + sensibilidad);

            System.out.println("\n=== RESULTADOS ===");
            System.out.printf("Precisión: %.2f%n", precision);
            System.out.printf("Sensibilidad: %.2f%n", sensibilidad);
            System.out.printf("F1-score: %.2f%n%n", f1);

            System.out.println("=== CLASIFICACIÓN DE P1..P4 ===");
            Punto[] nuevos = datos.subList(0, 4).toArray(new Punto[0]);

            for (int i = 0; i < nuevos.length; i++) {
                System.out.println("P" + (i + 1) + " -> " + knn(train, nuevos[i], k));
            }
        }
    }
