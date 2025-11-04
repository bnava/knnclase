import smile.classification.KNN;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.vector.IntVector;
import smile.io.Read;
import smile.feature.Standardizer;
import smile.validation.ClassificationMetrics;

import java.nio.file.Paths;

public class KNNFlores {
    public static void main(String[] args) throws Exception {

        System.out.println("Cargando datos...");
        DataFrame datos = Read.csv(Paths.get("data/dataset_flores_ruido.csv"));
        System.out.println("Datos cargados correctamente.\n");

        double[][] X = datos.select("Largo_Petalo", "Ancho_Petalo").toArray();
        int[] y = datos.column("Tipo_Flor").toIntArray();

        Standardizer scaler = new Standardizer();
        double[][] X_scaled = scaler.fit(X).transform(X);

        int n = X_scaled.length;
        int trainSize = (int)(n * 0.7);
        double[][] X_train = new double[trainSize][];
        double[][] X_test = new double[n - trainSize][];
        int[] y_train = new int[trainSize];
        int[] y_test = new int[n - trainSize];

        for (int i = 0; i < n; i++) {
            if (i < trainSize) {
                X_train[i] = X_scaled[i];
                y_train[i] = y[i];
            } else {
                X_test[i - trainSize] = X_scaled[i];
                y_test[i - trainSize] = y[i];
            }
        }

        int k_vecinos = 3;
        KNN<double[]> knn = KNN.fit(X_train, y_train, k_vecinos);

        double[][] patronesEntrada = {
                {5.1235, 6.8832},
                {4.9842, 7.8921},
                {6.1213, 4.1235},
                {2.3123, 8.1231}
        };

        double[][] patronesEscalados = scaler.transform(patronesEntrada);
        int[] predicciones = new int[patronesEscalados.length];

        System.out.println("Clasificación de patrones de entrada:");
        for (int i = 0; i < patronesEscalados.length; i++) {
            predicciones[i] = knn.predict(patronesEscalados[i]);
            System.out.printf("P%d: Largo=%.4f, Ancho=%.4f -> Clase=%d%n",
                    i + 1, patronesEntrada[i][0], patronesEntrada[i][1], predicciones[i]);
        }

        int[] y_pred = new int[y_test.length];
        for (int i = 0; i < y_test.length; i++) {
            y_pred[i] = knn.predict(X_test[i]);
        }

        ClassificationMetrics metrics = ClassificationMetrics.of(y_test, y_pred);
        System.out.println("\n--- MÉTRICAS DE EVALUACIÓN ---");
        System.out.println("Accuracy: " + metrics.accuracy);
        System.out.println("Precision: " + metrics.precision);
        System.out.println("Recall: " + metrics.recall);
        System.out.println("F1: " + metrics.f1);
    }
}
