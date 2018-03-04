import javax.swing.*;
import java.awt.*;

public class PerceptronLearning {
    //---------------Constants----------------------//
    public static final int numberOfHiddenNodes = 100;
    public static final int train_total = 60000;
    public static final int test_total = 10000;
    public static final double learningRate = 0.1;
    public static final int maxNumberOfEpochs = 500;
    public static final double momentum = 0.9;

    private double[][] weights_input_hidden = new double[numberOfHiddenNodes][785];
    private double[][] weights_hidden_output = new double[10][numberOfHiddenNodes];

    Label_Input train_label_input;
    int[] train_label;
    double[][] train_input;
    double[] train_accuracy = new double[maxNumberOfEpochs + 1]; // +1 to store initial accuracy before training

    Label_Input test_label_input;
    int[] test_label;
    double[][] test_input;
    double[] test_accuracy = new double[maxNumberOfEpochs + 1]; // +1 to store initial accuracy before training

    public void perceptronLearning() {
        //----------Get data for training-----------------//
        train_label_input = Accuracy.getData("train");
        train_label = train_label_input.getLabel();
        train_input = train_label_input.getInput();

        //----------Get data for testing-----------------//
        test_label_input = Accuracy.getData("test");
        test_label = test_label_input.getLabel();
        test_input = test_label_input.getInput();

        //-----------Get Initial Weights-------------------//
        weights_input_hidden = Accuracy.generateRandomWeights(numberOfHiddenNodes, 785);
        weights_hidden_output = Accuracy.generateRandomWeights(10, numberOfHiddenNodes);

        // Store the initial accuracy before training
        train_accuracy[0] = findAccuracy("train");
        test_accuracy[0] = findAccuracy("test");

        // Adjust the weights if the output is not same as expected.
        // For each epoch of training, find accuracy.
        // Repeat this for maxNumberOfEpochs
        for (int n = 1; n <= maxNumberOfEpochs; n++) {
            adjustWeights();

            train_accuracy[n] = findAccuracy("train");
            test_accuracy[n] = findAccuracy("test");
        }
    }

    /**
     * Find the accuracy = number of correct classification/ total number of classifications
     */
    private double findAccuracy(String type) {
        double correct;
        double accuracy;
        if (type.equals("train")) {
            correct = Accuracy.findClassifications(true, train_total, weights_input_hidden, weights_hidden_output, train_label, train_input);
            accuracy = correct * 100 / train_total;
        } else {
            correct = Accuracy.findClassifications(true, test_total, weights_input_hidden, weights_hidden_output, test_label, test_input);
            accuracy = correct * 100 / test_total;
        }
        return accuracy;
    }

    /**
     * Adjust the weights using back propagation if the output is not equal to the expected output
     */
    private void adjustWeights() {
        Accuracy.findClassifications(false, train_total, weights_input_hidden, weights_hidden_output, train_label, train_input);
    }

    /**
     * Plot the accuracy graphs with epoch as X-axis and accuracy percentage as Y-axis for both training and test data
     */
    public void plotAccuracyGraphs() {
        PlotGraph plotGraph = new PlotGraph("Accuracy Plots", train_accuracy, test_accuracy);
        plotGraph.pack();
        plotGraph.setVisible(true);
    }

    /**
     * Confusion matrix for the test data after the model is trained for maximum number of epochs.
     */
    public void showConfusionMatrix() {
        JTable table = new JTable(11, 11);
        table.setDefaultRenderer(Object.class, new CustomTableCellRenderer());

        table.setShowGrid(true);
        table.setGridColor(Color.BLACK);

        for (int i = 1; i < table.getColumnCount(); i++) {
            table.setValueAt(i - 1, 0, i);
        }

        for (int i = 1; i < table.getRowCount(); i++) {
            table.setValueAt(i - 1, i, 0);
        }

        int[] outputs = Accuracy.findClassificationsForConfusionMatrix(weights_input_hidden, weights_hidden_output, test_input);
        int[][] grid = new int[10][10];

        for (int n = 0; n < 10000; n++) {
            grid[test_label[n]][outputs[n]]++;
        }

        for (int r = 1; r < table.getRowCount(); r++) {
            for (int c = 1; c < table.getColumnCount(); c++) {
                table.setValueAt(grid[r - 1][c - 1], r, c);
            }
        }

        JFrame frame = new JFrame("Confusion Matrix");
        frame.getContentPane().add(table);
        frame.pack();
        frame.setVisible(true);
    }
}
