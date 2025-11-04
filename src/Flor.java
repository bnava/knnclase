public class Flor {
    private double sepalLength;
    private double sepalWidth;
    private double petalLength;
    private double petalWidth;
    private String species;
    
    public Flor(double sepalLength, double sepalWidth, double petalLength, double petalWidth, String species) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        this.species = species;
    }
    
    public double getSepalLength() { return sepalLength; }
    public double getSepalWidth() { return sepalWidth; }
    public double getPetalLength() { return petalLength; }
    public double getPetalWidth() { return petalWidth; }
    public String getSpecies() { return species; }
    
    public double calcularDistancia(Flor otra) {
        return Math.sqrt(
            Math.pow(this.sepalLength - otra.sepalLength, 2) +
            Math.pow(this.sepalWidth - otra.sepalWidth, 2) +
            Math.pow(this.petalLength - otra.petalLength, 2) +
            Math.pow(this.petalWidth - otra.petalWidth, 2)
        );
    }
    
    @Override
    public String toString() {
        return String.format("Flor{SL=%.2f, SW=%.2f, PL=%.2f, PW=%.2f, Species='%s'}", 
                           sepalLength, sepalWidth, petalLength, petalWidth, species);
    }
}