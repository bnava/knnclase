package mx.uaem;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Knn_Hugo {

    // Parámetros del programa
    static final String XLSX_PATH = "dataset_flores_ruido.xlsx";
    static final int K = 3;

    record Punto(double largo, double ancho, String clase) {}
    record Escala(double media, double std) {}

    public static void main(String[] args) throws Exception {

        // 1) Validar archivo
        if (!Files.exists(Path.of(XLSX_PATH))) {
            System.out.println("ERROR: Coloca el archivo dataset_flores_ruido.xlsx junto al pom.xml");
            return;
        }

        // 2) Leer datos
        List<Punto> datos = leerXLSX(XLSX_PATH);
        System.out.println("Datos cargados: " + datos.size());

        // 3) Mezclar y dividir en train/test
        Collections.shuffle(datos, new Random(42));
        int corte = (int) (datos.size() * 0.30);
        List<Punto> test = datos.subList(0, corte);
        List<Punto> train = datos.subList(corte, datos.size());

        // 4) Calcular medias/STD usando solo train
        Escala sL = calcularEscala(train, true);
        Escala sA = calcularEscala(train, false);

        List<Punto> trainZ = estandarizar(train, sL, sA);
        List<Punto> testZ  = estandarizar(test,  sL, sA);

        // 5) Evaluar KNN (matriz, accuracy, precision, recall, F1)
        evaluarKNN(trainZ, testZ);

        // 6) P1–P4
        double[][] nuevos = {
                {5.1235, 6.8832},
                {4.9842, 7.8921},
                {6.1213, 4.1235},
                {2.3123, 8.1231}
        };
        System.out.println("\nPREDICCIONES:");
        for (int i = 0; i < nuevos.length; i++) {
            double zl = (nuevos[i][0] - sL.media) / sL.std;
            double za = (nuevos[i][1] - sA.media) / sA.std;
            String clase = knn(trainZ, zl, za, K);
            System.out.printf("P%d  (%.4f, %.4f) → %s%n", i + 1, nuevos[i][0], nuevos[i][1], clase);
        }

        // 7) Gráfica
        mostrarGrafica(trainZ);
    }

    // Leer XLSX
    static List<Punto> leerXLSX(String archivo) throws Exception {
        List<Punto> lista = new ArrayList<>();

        try (FileInputStream fis = new FileInputStream(archivo);
             Workbook wb = new XSSFWorkbook(fis)) {

            Sheet hoja = wb.getSheetAt(0);
            Iterator<Row> it = hoja.iterator();
            Row cab = it.next(); // leer encabezados

            int cL = -1, cA = -1, cC = -1;
            for (Cell cel : cab) {
                String n = cel.getStringCellValue().trim();
                if (n.equals("Largo_Petalo")) cL = cel.getColumnIndex();
                if (n.equals("Ancho_Petalo")) cA = cel.getColumnIndex();
                if (n.equals("Tipo_Flor"))    cC = cel.getColumnIndex();
            }

            while (it.hasNext()) {
                Row r = it.next();
                double largo = r.getCell(cL).getNumericCellValue();
                double ancho = r.getCell(cA).getNumericCellValue();
                String clase = r.getCell(cC).getStringCellValue().trim();
                lista.add(new Punto(largo, ancho, clase));
            }
        }
        return lista;
    }

    // Calcular media y desviación estándar
    static Escala calcularEscala(List<Punto> lista, boolean usarLargo){
        double suma = 0;
        for (Punto p : lista) suma += (usarLargo ? p.largo : p.ancho);
        double media = suma / lista.size();

        double var = 0;
        for (Punto p : lista){
            double v = usarLargo ? p.largo : p.ancho;
            var += (v - media) * (v - media);
        }
        var /= Math.max(1, (lista.size() - 1)); // evitar división por cero
        return new Escala(media, Math.sqrt(Math.max(var, 1e-12)));
    }

    // Z-score
    static List<Punto> estandarizar(List<Punto> lista, Escala sL, Escala sA){
        List<Punto> res = new ArrayList<>();
        for (Punto p : lista){
            double zl = (p.largo - sL.media) / sL.std;
            double za = (p.ancho - sA.media) / sA.std;
            res.add(new Punto(zl, za, p.clase));
        }
        return res;
    }

    // KNN simple
    static String knn(List<Punto> train, double x, double y, int k){
        List<Punto> copia = new ArrayList<>(train);
        copia.sort(Comparator.comparingDouble(p -> Math.hypot(p.largo - x, p.ancho - y)));

        Map<String,Integer> votos = new HashMap<>();
        int limite = Math.min(k, copia.size());
        for (int i = 0; i < limite; i++){
            String c = copia.get(i).clase;
            votos.put(c, votos.getOrDefault(c, 0) + 1);
        }

        String mejor = null;
        int maxVotos = -1;
        for (Map.Entry<String,Integer> e : votos.entrySet()){
            if (e.getValue() > maxVotos){
                mejor = e.getKey();
                maxVotos = e.getValue();
            }
        }
        return mejor;
    }

    // === Métricas completas: matriz de confusión, accuracy, precisión, recall y F1 ===
    static void evaluarKNN(List<Punto> trainZ, List<Punto> testZ){
        // Orden de clases según aparecen en test (sin streams)
        List<String> clases = new ArrayList<>();
        for (Punto p : testZ) if (!clases.contains(p.clase)) clases.add(p.clase);

        int C = clases.size();
        Map<String,Integer> idx = new HashMap<>();
        for (int i = 0; i < C; i++) idx.put(clases.get(i), i);

        int[][] cm = new int[C][C]; // [real][predicha]
        int correctos = 0;

        // Construir matriz de confusión
        for (Punto p : testZ){
            String pred = knn(trainZ, p.largo, p.ancho, K);
            if (pred.equals(p.clase)) correctos++;
            cm[idx.get(p.clase)][idx.get(pred)]++;
        }

        double accuracy = correctos / (double) testZ.size();

        // Imprimir matriz
        System.out.println("\nMatriz de Confusión (filas=real, columnas=pred):");
        System.out.print("           ");
        for (String c : clases) System.out.printf("%10s", c);
        System.out.println();
        for (int i = 0; i < C; i++){
            System.out.printf("%10s", clases.get(i));
            for (int j = 0; j < C; j++){
                System.out.printf("%10d", cm[i][j]);
            }
            System.out.println();
        }

        // Métricas por clase
        double macroP = 0, macroR = 0, macroF1 = 0;
        System.out.println("\n== Métricas por clase ==");
        for (int k = 0; k < C; k++){
            int TP = cm[k][k];
            int FP = 0, FN = 0;
            for (int i = 0; i < C; i++){
                if (i != k){
                    FP += cm[i][k]; // predije k pero era otra
                    FN += cm[k][i]; // era k pero predije otra
                }
            }
            double precision = (TP + FP) == 0 ? 0 : TP / (double)(TP + FP);
            double recall    = (TP + FN) == 0 ? 0 : TP / (double)(TP + FN);
            double f1        = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

            macroP  += precision;
            macroR  += recall;
            macroF1 += f1;

            System.out.printf("%s -> precision=%.2f  recall=%.2f  f1=%.2f%n",
                    clases.get(k), precision, recall, f1);
        }
        macroP  /= C;
        macroR  /= C;
        macroF1 /= C;

        System.out.printf("%nAccuracy=%.2f  |  Macro-Precision=%.2f  Macro-Recall=%.2f  Macro-F1=%.2f%n",
                accuracy, macroP, macroR, macroF1);
    }

    // Gráfica
    static void mostrarGrafica(List<Punto> lista){
        XYChart c = new XYChartBuilder().width(900).height(600)
                .title("KNN Flores")
                .xAxisTitle("Largo(z)")
                .yAxisTitle("Ancho(z)")
                .build();

        Map<String,List<Punto>> porClase = new LinkedHashMap<>();
        for (Punto p : lista){
            porClase.computeIfAbsent(p.clase, v -> new ArrayList<>()).add(p);
        }

        for (Map.Entry<String,List<Punto>> e : porClase.entrySet()){
            List<Double> xs = new ArrayList<>(), ys = new ArrayList<>();
            for (Punto p : e.getValue()){
                xs.add(p.largo); ys.add(p.ancho);
            }
            var s = c.addSeries(e.getKey(), xs, ys);
            s.setMarker(SeriesMarkers.CIRCLE);
        }

        new SwingWrapper<>(c).displayChart();
        System.out.println("Gráfica abierta ");
    }
}
