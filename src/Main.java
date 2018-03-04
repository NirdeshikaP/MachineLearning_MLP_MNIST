/**
 * The main driver class which calls the appropriate sections of the code to train the model,
 * plot the accuracy graphs and to show the confusion matrix.
 */
public class Main {
    public static void main(String[] args)
    {
        PerceptronLearning perceptronLearning = new PerceptronLearning();
        long startTime = System.currentTimeMillis();
        perceptronLearning.perceptronLearning();
        long endTime = System.currentTimeMillis();
        System.out.println(endTime-startTime);
        perceptronLearning.plotAccuracyGraphs();
        perceptronLearning.showConfusionMatrix();
    }
}
