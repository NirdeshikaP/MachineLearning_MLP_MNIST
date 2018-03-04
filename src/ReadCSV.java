import java.io.*;

/**
 * To read data from the CSV files, parse it and store it in arrays
 */
public class ReadCSV {
    private String fileName = null;
    private int[] label ;
    private double[][] input;

    public ReadCSV(String fileName, String type){
        this.fileName = fileName;
        if(type.equals("train"))
        {
            label = new int[PerceptronLearning.train_total];
            input = new double[PerceptronLearning.train_total][785];
        }
        else
        {
            label = new int[PerceptronLearning.test_total];
            input = new double[PerceptronLearning.test_total][785];
        }
    }
    /**
     * To read data from the CSV files, parse it and store it in arrays
     */
    public Label_Input readCSV(){
        try {
            FileReader fileReader = new FileReader(fileName);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            int i = 0;

            int v = 0;
            while((line = bufferedReader.readLine()) != null){
                String[] values = line.split(",");

                label[i] = Integer.parseInt(values[0]);
                input[i][0] = 1; // Bias
                for(int k = 1; k<values.length; k++)
                {
                    v = Integer.parseInt(values[k]);
                    input[i][k] =  (double)v / 255; // normalize
                }
                i++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new Label_Input(label,input);
    }
}
