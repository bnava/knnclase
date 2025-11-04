import java.util.*;

public class KNN {
    private List<Flor> dataset;
    private int k;
    
    public KNN(List<Flor> dataset, int k) {
        this.dataset = dataset;
        this.k = k;
    }
    
    public String clasificar(Flor nuevaFlor) {
        List<VecinoDistancia> vecinos = new ArrayList<>();
        
        for (Flor flor : dataset) {
            double distancia = nuevaFlor.calcularDistancia(flor);
            vecinos.add(new VecinoDistancia(flor, distancia));
        }
        
        vecinos.sort(Comparator.comparingDouble(VecinoDistancia::getDistancia));
        
        Map<String, Integer> votos = new HashMap<>();
        for (int i = 0; i < k && i < vecinos.size(); i++) {
            String especie = vecinos.get(i).getFlor().getSpecies();
            votos.put(especie, votos.getOrDefault(especie, 0) + 1);
        }
        
        return votos.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("Desconocida");
    }
    
    private static class VecinoDistancia {
        private Flor flor;
        private double distancia;
        
        public VecinoDistancia(Flor flor, double distancia) {
            this.flor = flor;
            this.distancia = distancia;
        }
        
        public Flor getFlor() { return flor; }
        public double getDistancia() { return distancia; }
    }
}