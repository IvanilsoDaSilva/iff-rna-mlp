import java.util.Random;

class MLP {
    static private double[][] weightsInputHidden;
    static private double[][] weightsHiddenOutput;
    private double learningRate;

    public MLP(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        this.learningRate = learningRate;

        initializeWeights(weightsInputHidden);
        initializeWeights(weightsHiddenOutput);
    }

    private void initializeWeights(double[][] weights) {
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = random.nextDouble() * 0.1;  // Inicialização dos pesos
            }
        }
    }

    private double dotProduct(double[] vectorA, double[] vectorB) {
        double result = 0;
        for (int i = 0; i < vectorA.length; i++) {
            result += vectorA[i] * vectorB[i];
        }
        return result;
    }

    private double[] getColumn(double[][] matrix, int columnIndex) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][columnIndex];
        }
        return column;
    }

    static private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    static public double[] feedForward(double[] input) {
        double[] hiddenLayer = new double[weightsInputHidden[0].length];
        double[] outputLayer = new double[weightsHiddenOutput[0].length];

        // Calculando a saída da camada oculta
        for (int j = 0; j < hiddenLayer.length; j++) {
            hiddenLayer[j] = 0;
            for (int i = 0; i < input.length; i++) {
                hiddenLayer[j] += input[i] * weightsInputHidden[i][j];
            }
            hiddenLayer[j] = sigmoid(hiddenLayer[j]);
        }

        // Calculando a saída da camada de saída
        for (int k = 0; k < outputLayer.length; k++) {
            outputLayer[k] = 0;
            for (int j = 0; j < hiddenLayer.length; j++) {
                outputLayer[k] += hiddenLayer[j] * weightsHiddenOutput[j][k];
            }
            outputLayer[k] = sigmoid(outputLayer[k]);
        }

        return outputLayer;
    }

    public void train(double[][] inputs, double[][] expectedOutputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double[] expected = expectedOutputs[i];

                // Forward pass
                double[] hiddenLayer = new double[weightsInputHidden[0].length];
                double[] output = feedForward(input);

                // Backpropagation para atualizar os pesos
                // (Aqui deverá ter a implementação da atualização de pesos com base no erro)
            }
        }
    }
}