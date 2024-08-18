public class Main {
    public static void main(String[] args) {
        // Valores de entrada
        int inputSize = 3;
        int hiddenSize = 3;
        int outputSize = 3;
        int epochs = 3;

        // Criando uma instância da classe MLP
        MLP mlp = new MLP(inputSize, hiddenSize, outputSize, epochs);

        // Exemplo de entrada
        double[] input = {0.2, 0.8, 0.2}; // Exemplo de dados de entrada

        // Chamando o método feedForward
        double[] output = mlp.feedForward(input);

        // Imprimindo a saída
        System.out.println("Saída da rede MLP:");
        for (double o : output) {
            System.out.println(o);
        }
    }
}