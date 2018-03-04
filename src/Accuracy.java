import java.util.Random;
import java.util.stream.DoubleStream;

/**
 * The helper class for PerceptronLearning.
 */
public class Accuracy {
    static double[][] changeInWeightsOfHiddenNodes = new double[PerceptronLearning.numberOfHiddenNodes][785];
    static double[][] changeInWeightsOfOutputNodes = new double[10][PerceptronLearning.numberOfHiddenNodes];

    /**
     * Read data from CSV files
     */
    public static Label_Input getData(String type) {
        ReadCSV r;
        if (type.equals("train")) {
            r = new ReadCSV("mnist_train.csv", type);
        } else
            r = new ReadCSV("mnist_test.csv", type);
        return r.readCSV();
    }

    /**
     * Generate values for weights with random values between -0.05 and 0.05
     */
    public static double[][] generateRandomWeights(int rows, int columns) {
        Random random = new Random();
        DoubleStream doubleStream;
        double[][] weights = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            doubleStream = random.doubles(columns, -0.05, 0.05);
            weights[i] = doubleStream.toArray();
        }
        return weights;
    }

    /**
     * 1. Forward propagation: Find values of hidden nodes and output nodes
     * 2. Find errors
     * 3. Backward propagation: adjust the weights
     *  If isForAccuracy is true, weights will not be adjusted. Else, weights are adjusted.
     */
    public static int findClassifications(boolean isForAccuracy, int total, double[][] weights_layer1, double[][] weights_layer2, int[] label, double[][] input) {
        int correct = 0;

        for (int i = 0; i < total; i++) { //For each example
            double[] hiddenNodeValue = new double[PerceptronLearning.numberOfHiddenNodes];
            double[] outputValue = new double[10];
            for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) { //Calculate the value for each hidden neuron
                for (int j = 0; j < 785; j++) {
                    hiddenNodeValue[h] += weights_layer1[h][j] * input[i][j];
                }
                hiddenNodeValue[h] = 1 / (1 + Math.pow(Math.E, -hiddenNodeValue[h])); // Passing through activation function, sigmoid
            }
            for (int o = 0; o < 10; o++) { //Calculate the output for each neuron
                for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) {
                    outputValue[o] += weights_layer2[o][h] * hiddenNodeValue[h];
                }
                outputValue[o] = 1 / (1 + Math.pow(Math.E, -outputValue[o])); // Passing through activation function, sigmoid
            }
            if (isForAccuracy) {
                if (label[i] == findMaximum(outputValue))
                    correct++;
            } else if (!isForAccuracy) {
                ErrorValues errorValues = calculateErrors(outputValue, hiddenNodeValue, label[i], weights_layer2);
                updateWeights(errorValues, weights_layer1, weights_layer2, hiddenNodeValue, input[i]);
            }
        }
        return correct;
    }

    /**
     * Adjust the weights
     */
    private static void updateWeights(ErrorValues errorValues, double[][] weights_hiddenLayer, double[][] weights_outputLayer, double[] hiddenNodeValue, double[] input) {
        double[] hiddenLayerErrorValues = errorValues.getHiddenLayerErrorValues();
        double[] outputLayerErrorValues = errorValues.getOutputLayerErrorValues();

        // Adjust the weights of the layer between hidden nodes and output nodes
        for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) {
            for (int o = 0; o < 10; o++) {
                changeInWeightsOfOutputNodes[o][h] = PerceptronLearning.learningRate * outputLayerErrorValues[o] * hiddenNodeValue[h]
                        + PerceptronLearning.momentum * changeInWeightsOfOutputNodes[o][h];

                weights_outputLayer[o][h] += changeInWeightsOfOutputNodes[o][h];
            }
        }
        // Adjust the weights of the layer between hidden nodes and input nodes
        for (int i = 0; i < 785; i++) {
            for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) {
                changeInWeightsOfHiddenNodes[h][i] = PerceptronLearning.learningRate * hiddenLayerErrorValues[h] * input[i]
                        + PerceptronLearning.momentum * changeInWeightsOfHiddenNodes[h][i];

                weights_hiddenLayer[h][i] += changeInWeightsOfHiddenNodes[h][i];
            }
        }
    }

    /**
     * 1. Compute the errors at the outputs
     * 2. Compute the errors at the hidden layers
     */
    private static ErrorValues calculateErrors(double[] outputValue, double[] hiddenNodeValue, int label, double[][] weights_outputLayer) {
        double[] outputErrors = new double[10];
        double[] hiddenLayerErrors = new double[PerceptronLearning.numberOfHiddenNodes];

        for (int o = 0; o < 10; o++) {
            double y = outputValue[o];
            double t = label == o ? 0.9 : 0.1;
            outputErrors[o] = (t - y) * y * (1 - y);
        }

        for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) {
            double a = hiddenNodeValue[h];
            double productOfWeights = 0.0;
            for (int o = 0; o < 10; o++) {
                productOfWeights += (outputErrors[o] * weights_outputLayer[o][h]);
            }
            hiddenLayerErrors[h] = a * (1 - a) * productOfWeights;
        }

        return new ErrorValues(hiddenLayerErrors, outputErrors);
    }

    /**
     * Find the correct classification of test data for showing the confusion matrix
     */
    public static int[] findClassificationsForConfusionMatrix(double[][] weights_layer1, double[][] weights_layer2, double[][] input) {
        int[] output = new int[PerceptronLearning.test_total];

        for (int i = 0; i < PerceptronLearning.test_total; i++) { //For each example
            double[] hiddenNodeValue = new double[PerceptronLearning.numberOfHiddenNodes];
            double[] outputValue = new double[10];
            for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) { //Calculate the value for each hidden neuron
                for (int j = 0; j < 785; j++) {
                    hiddenNodeValue[h] += weights_layer1[h][j] * input[i][j];
                }
                hiddenNodeValue[h] = 1 / (1 + Math.pow(Math.E, -hiddenNodeValue[h])); // Passing through activation function, sigmoid
            }
            for (int o = 0; o < 10; o++) { //Calculate the output for each neuron
                for (int h = 0; h < PerceptronLearning.numberOfHiddenNodes; h++) {
                    outputValue[o] += weights_layer2[o][h] * hiddenNodeValue[h];
                }
                outputValue[o] = 1 / (1 + Math.pow(Math.E, -outputValue[o])); // Passing through activation function, sigmoid
            }

            output[i] = findMaximum(outputValue);
        }
        return output;
    }

    /**
     * This is to find the output value of the model. The neuron with the maximum value is the output.
     * For example, if node 0 has the maximum value, then the output is 0.
     */
    private static int findMaximum(double[] a) {
        double maximumValue = a[0];
        int maximumIndex = 0;

        for (int i = 1; i < a.length; i++) {
            if (maximumValue < a[i]) {
                maximumIndex = i;
                maximumValue = a[i];
            }
        }
        return maximumIndex;
    }
}
