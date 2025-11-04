import java.util.List;
import java.util.Scanner;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        compilarArchivos();
        
        try {
            String rutaArchivo = "dataset_flores_ruido.xlsx";
            List<Flor> dataset = ExcelReader.leerDataset(rutaArchivo);
            
            if (dataset.isEmpty()) {
                System.out.println("No se pudieron cargar datos del archivo.");
                return;
            }
            
            System.out.println("Dataset cargado: " + dataset.size() + " flores");
            
            Scanner scanner = new Scanner(System.in);
            System.out.print("Ingrese el valor de K: ");
            int k = scanner.nextInt();
            
            KNN knn = new KNN(dataset, k);
            
            System.out.println("\nIngrese los datos de la nueva flor:");
            System.out.print("Sepal Length: ");
            double sepalLength = scanner.nextDouble();
            
            System.out.print("Sepal Width: ");
            double sepalWidth = scanner.nextDouble();
            
            System.out.print("Petal Length: ");
            double petalLength = scanner.nextDouble();
            
            System.out.print("Petal Width: ");
            double petalWidth = scanner.nextDouble();
            
            Flor nuevaFlor = new Flor(sepalLength, sepalWidth, petalLength, petalWidth, "");
            
            String clasificacion = knn.clasificar(nuevaFlor);
            
            System.out.println("\nResultado:");
            System.out.println("Flor clasificada como: " + clasificacion);
            System.out.println("Datos ingresados: " + nuevaFlor);
            
            scanner.close();
            
        } catch (Exception e) {
            System.err.println("Error en la aplicación: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void compilarArchivos() {
        try {
            Process proceso = Runtime.getRuntime().exec("javac *.java");
            proceso.waitFor();
            if (proceso.exitValue() == 0) {
                System.out.println("Compilación exitosa");
            } else {
                System.out.println("Error en compilación");
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Error compilando: " + e.getMessage());
        }
    }
}