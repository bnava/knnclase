import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class ExcelReader {
    private static final Pattern NUMERO_PATTERN = Pattern.compile("\\d+\\.?\\d*");
    
    public static List<Flor> leerDataset(String rutaArchivo) {
        List<Flor> flores = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(rutaArchivo))) {
            String linea;
            boolean primeraLinea = true;
            
            while ((linea = reader.readLine()) != null) {
                if (primeraLinea) {
                    primeraLinea = false;
                    continue;
                }
                
                try {
                    Flor flor = parsearLinea(linea);
                    if (flor != null) {
                        flores.add(flor);
                    }
                } catch (Exception e) {
                    System.err.println("Error procesando línea: " + linea + " - " + e.getMessage());
                }
            }
            
        } catch (IOException e) {
            System.err.println("Error leyendo archivo: " + e.getMessage());
        }
        
        return flores;
    }
    
    private static Flor parsearLinea(String linea) {
        String[] partes = linea.split("[,;\\t]");
        
        if (partes.length < 5) {
            return null;
        }
        
        try {
            double sepalLength = extraerNumero(partes[0]);
            double sepalWidth = extraerNumero(partes[1]);
            double petalLength = extraerNumero(partes[2]);
            double petalWidth = extraerNumero(partes[3]);
            String species = partes[4].trim().replaceAll("\"", "");
            
            return new Flor(sepalLength, sepalWidth, petalLength, petalWidth, species);
            
        } catch (NumberFormatException e) {
            return null;
        }
    }
    
    private static double extraerNumero(String texto) {
        if (NUMERO_PATTERN.matcher(texto.trim()).matches()) {
            return Double.parseDouble(texto.trim());
        }
        
        String numeroLimpio = texto.replaceAll("[^\\d.]", "");
        return Double.parseDouble(numeroLimpio);
    }
}