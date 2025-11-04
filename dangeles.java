import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultXYDataset;

import java.io.FileInputStream;
import java.util.*;
import java.io.File;

public class dangeles {
    public static void main(String[] args) {
        System.out.println("=== ALGORITMO KNN - CLASIFICACIÓN DE FLORES ===");

        // Configuración de rutas - prueba con diferentes ubicaciones
        String[] rutasPosibles = {
                "C:\\Users\\dania\\IdeaProjects\\knn\\src\\main\\java\\dataset_flores_ruido.xlsx"
        };

        String rutaArchivo = null;
        File archivo = null;

        // Buscar el archivo en diferentes ubicaciones
        for (String ruta : rutasPosibles) {
            archivo = new File(ruta);
            if (archivo.exists()) {
                rutaArchivo = ruta;
                break;
            }
        }

        if (rutaArchivo == null) {
            System.out.println("Error: No se encuentra el archivo flores.xlsx");
            for (String ruta : rutasPosibles) {
                System.out.println("   - " + ruta);
            }
            return;
        }
        ExcelReader reader = new ExcelReader();
        List<Flor> flores = reader.leerArchivoExcel(rutaArchivo);

        if (flores.isEmpty()) {
            System.out.println("No se pudieron leer datos del archivo Excel.");
            return;
        }
        for (int i = 0; i < Math.min(7, flores.size()); i++) {
            Flor flor = flores.get(i);
            System.out.printf("%-3d %-11.4f %-11.4f %-10s %n",
                    i+1, flor.getLargoPetalo(), flor.getAnchoPetalo(), flor.getTipoFlor());
        }
        List<Flor> patronesPrueba = Arrays.asList(
                new Flor(5.1235, 6.8832, "Por predecir"),
                new Flor(4.9842, 7.8921, "Por predecir"),
                new Flor(6.1213, 4.1235, "Por predecir"),
                new Flor(2.3123, 8.1231, "Por predecir")
        );

        ejecutarAlgoritmoKNN(flores, patronesPrueba);
    }

    private static void ejecutarAlgoritmoKNN(List<Flor> flores, List<Flor> patronesPrueba) {
        System.out.println(flores.size());
        KNN knn = new KNN(3, flores);
        for (int i = 0; i < patronesPrueba.size(); i++) {
            Flor p = patronesPrueba.get(i);
            String prediccion = knn.predecir(p.getLargoPetalo(), p.getAnchoPetalo());
            System.out.printf("P%d: (%.4f, %.4f) → %s%n",
                    i+1, p.getLargoPetalo(), p.getAnchoPetalo(), prediccion);
        }
    }
}
class Flor {
    private double largoPetalo;
    private double anchoPetalo;
    private String tipoFlor;
    private String tipoPredicho; // Para almacenar la predicción

    public Flor(double largoPetalo, double anchoPetalo, String tipoFlor) {
        this.largoPetalo = largoPetalo;
        this.anchoPetalo = anchoPetalo;
        this.tipoFlor = tipoFlor;
    }

    // Getters y Setters
    public double getLargoPetalo() { return largoPetalo; }
    public void setLargoPetalo(double largoPetalo) { this.largoPetalo = largoPetalo; }

    public double getAnchoPetalo() { return anchoPetalo; }
    public void setAnchoPetalo(double anchoPetalo) { this.anchoPetalo = anchoPetalo; }

    public String getTipoFlor() { return tipoFlor; }
    public void setTipoFlor(String tipoFlor) { this.tipoFlor = tipoFlor; }

    public String getTipoPredicho() { return tipoPredicho; }
    public void setTipoPredicho(String tipoPredicho) { this.tipoPredicho = tipoPredicho; }

    @Override
    public String toString() {
        return String.format("Flor{Largo=%.4f, Ancho=%.4f, Tipo=%s, Predicho=%s}",
                largoPetalo, anchoPetalo, tipoFlor, tipoPredicho);
    }
}
class ExcelReader {
    public List<Flor> leerArchivoExcel(String rutaArchivo) {
        List<Flor> flores = new ArrayList<>();

        try (FileInputStream file = new FileInputStream(new File(rutaArchivo));
             Workbook workbook = new XSSFWorkbook(file)) {

            Sheet sheet = workbook.getSheetAt(0); // Primera hoja

            // Encontrar índices de columnas
            int indiceLargo = -1, indiceAncho = -1, indiceTipo = -1;
            Row headerRow = sheet.getRow(0);

            for (Cell cell : headerRow) {
                String header = cell.getStringCellValue().trim();
                if (header.equalsIgnoreCase("Largo_Petalo")) {
                    indiceLargo = cell.getColumnIndex();
                } else if (header.equalsIgnoreCase("Ancho_Petalo")) {
                    indiceAncho = cell.getColumnIndex();
                } else if (header.equalsIgnoreCase("Tipo_Flor")) {
                    indiceTipo = cell.getColumnIndex();
                }
            }

            if (indiceLargo == -1 || indiceAncho == -1 || indiceTipo == -1) {
                System.out.println("Error: No se encontraron todas las columnas necesarias");
                return flores;
            }

            // Leer datos
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                if (row != null) {
                    try {
                        double largoPetalo = row.getCell(indiceLargo).getNumericCellValue();
                        double anchoPetalo = row.getCell(indiceAncho).getNumericCellValue();
                        String tipoFlor = row.getCell(indiceTipo).getStringCellValue();

                        flores.add(new Flor(largoPetalo, anchoPetalo, tipoFlor));
                    } catch (Exception e) {
                        System.out.println("Error en fila " + (i+1) + ": " + e.getMessage());
                    }
                }
            }
            System.out.println("Se leyeron " + flores.size() + " registros del archivo Excel");
        } catch (Exception e) {
            System.out.println("Error al leer el archivo Excel: " + e.getMessage());
            e.printStackTrace();
        }

        return flores;
    }
}
class KNN {
    private int k;
    private List<Flor> datosEntrenamiento;
    public KNN(int k, List<Flor> datosEntrenamiento) {
        this.k = k;
        this.datosEntrenamiento = datosEntrenamiento;
    }
    public String predecir(double largoPetalo, double anchoPetalo) {
        // Lista para almacenar distancias y tipos
        List<DistanciaTipo> distancias = new ArrayList<>();

        // Calcular distancia a cada punto de entrenamiento
        for (Flor flor : datosEntrenamiento) {
            double distancia = calcularDistanciaEuclidiana(
                    largoPetalo, anchoPetalo,
                    flor.getLargoPetalo(), flor.getAnchoPetalo()
            );
            distancias.add(new DistanciaTipo(distancia, flor.getTipoFlor()));
        }

        // Ordenar por distancia (más cercanos primero)
        distancias.sort(Comparator.comparingDouble(DistanciaTipo::getDistancia));

        // Contar votos de los k vecinos más cercanos
        Map<String, Integer> votos = new HashMap<>();
        for (int i = 0; i < k && i < distancias.size(); i++) {
            String tipo = distancias.get(i).getTipo();
            votos.put(tipo, votos.getOrDefault(tipo, 0) + 1);
        }

        // Encontrar el tipo con más votos
        return Collections.max(votos.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    private double calcularDistanciaEuclidiana(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
    }

    // Clase auxiliar para almacenar distancia y tipo
    private class DistanciaTipo {
        private double distancia;
        private String tipo;

        public DistanciaTipo(double distancia, String tipo) {
            this.distancia = distancia;
            this.tipo = tipo;
        }

        public double getDistancia() { return distancia; }
        public String getTipo() { return tipo; }
    }

    // Método para calcular métricas de evaluación
    public Map<String, Double> calcularMetricas(List<Flor> datosTest) {
        int verdaderosPositivos = 0;
        int falsosPositivos = 0;
        int falsosNegativos = 0;
        int verdaderosNegativos = 0;

        Map<String, List<Integer>> metricasPorClase = new HashMap<>();

        // Inicializar estructura para cada tipo de flor
        Set<String> tipos = new HashSet<>();
        for (Flor flor : datosEntrenamiento) {
            tipos.add(flor.getTipoFlor());
        }

        for (String tipo : tipos) {
            metricasPorClase.put(tipo, Arrays.asList(0, 0, 0, 0)); // TP, FP, FN, TN
        }

        // Calcular predicciones y métricas
        for (Flor flor : datosTest) {
            String prediccion = predecir(flor.getLargoPetalo(), flor.getAnchoPetalo());
            flor.setTipoPredicho(prediccion);
            String real = flor.getTipoFlor();

            for (String tipo : tipos) {
                List<Integer> metrics = metricasPorClase.get(tipo);
                int tp = metrics.get(0);
                int fp = metrics.get(1);
                int fn = metrics.get(2);
                int tn = metrics.get(3);

                if (tipo.equals(real) && tipo.equals(prediccion)) {
                    tp++;
                } else if (tipo.equals(prediccion) && !tipo.equals(real)) {
                    fp++;
                } else if (tipo.equals(real) && !tipo.equals(prediccion)) {
                    fn++;
                } else {
                    tn++;
                }

                metricasPorClase.put(tipo, Arrays.asList(tp, fp, fn, tn));
            }
        }

        // Calcular métricas promedio
        double precisionTotal = 0;
        double sensibilidadTotal = 0;
        double f1ScoreTotal = 0;
        int count = 0;

        Map<String, Double> resultados = new HashMap<>();

        for (String tipo : tipos) {
            List<Integer> metrics = metricasPorClase.get(tipo);
            int tp = metrics.get(0);
            int fp = metrics.get(1);
            int fn = metrics.get(2);

            double precision = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
            double sensibilidad = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
            double f1Score = (precision + sensibilidad) == 0 ? 0 :
                    2 * (precision * sensibilidad) / (precision + sensibilidad);

            precisionTotal += precision;
            sensibilidadTotal += sensibilidad;
            f1ScoreTotal += f1Score;
            count++;

            resultados.put(tipo + "_precision", precision);
            resultados.put(tipo + "_sensibilidad", sensibilidad);
            resultados.put(tipo + "_f1score", f1Score);
        }
        // Métricas promedio
        resultados.put("precision_promedio", precisionTotal / count);
        resultados.put("sensibilidad_promedio", sensibilidadTotal / count);
        resultados.put("f1score_promedio", f1ScoreTotal / count);

        return resultados;
    }
}
class Grafico {
    public void mostrarDiagramaDispersion(List<Flor> flores, String titulo) {
        DefaultXYDataset dataset = new DefaultXYDataset();
        // Agrupar flores por tipo
        Map<String, List<double[]>> datosPorTipo = new HashMap<>();

        for (Flor flor : flores) {
            String tipo = flor.getTipoFlor();
            if (!datosPorTipo.containsKey(tipo)) {
                datosPorTipo.put(tipo, new java.util.ArrayList<>());
            }
            datosPorTipo.get(tipo).add(new double[]{
                    flor.getLargoPetalo(), flor.getAnchoPetalo()
            });
        }

        // Agregar series al dataset
        for (Map.Entry<String, List<double[]>> entry : datosPorTipo.entrySet()) {
            String tipo = entry.getKey();
            List<double[]> puntos = entry.getValue();

            double[][] serie = new double[2][puntos.size()];
            for (int i = 0; i < puntos.size(); i++) {
                serie[0][i] = puntos.get(i)[0]; // Largo_Petalo (X)
                serie[1][i] = puntos.get(i)[1]; // Ancho_Petalo (Y)
            }

            dataset.addSeries(tipo, serie);
        }

        // Crear el gráfico
        JFreeChart chart = ChartFactory.createScatterPlot(
                titulo,
                "Largo del Pétalo",
                "Ancho del Pétalo", dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Mostrar el gráfico en una ventana
        ChartFrame frame = new ChartFrame("Diagrama de Dispersión - KNN", chart);
        frame.pack();
        frame.setVisible(true);
    }
}