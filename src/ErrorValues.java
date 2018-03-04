/**
 * Class to hold the error values both input to hidden layer and also hidden to output
 */
public class ErrorValues {
    private double[] hiddenLayerErrorValues;
    private double[] outputLayerErrorValues;

    public ErrorValues(double[] hiddenLayerErrorValues, double[] outputLayerErrorValues) {
        this.hiddenLayerErrorValues = hiddenLayerErrorValues;
        this.outputLayerErrorValues = outputLayerErrorValues;
    }

    public double[] getHiddenLayerErrorValues() {
        return hiddenLayerErrorValues;
    }

    public double[] getOutputLayerErrorValues() {
        return outputLayerErrorValues;
    }
}
