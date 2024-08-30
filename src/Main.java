public class Main {
    public static void main(String[] args) {
        // Configurações da rede neural
        int tamanhoEntrada = 5; // Número de neurônios na camada de entrada
        int tamanhoOculta = 1;  // Número de neurônios na camada oculta
        int tamanhoSaida = 2;   // Número de neurônios na camada de saída
        double taxaAprendizado = 0.5; // Taxa de aprendizado
        int epocas = 9999; // Número de épocas de treinamento

        // Cria uma instância da rede neural MLP
        MLP mlp = new MLP(tamanhoEntrada, tamanhoOculta, tamanhoSaida, taxaAprendizado);

        // Dados de treinamento
        double[][] entradas = {
//              {bias, tosse, mancha na pele, corisa, baixa plaqueta}
                {1, 0, 1, 0},
                {0, 1, 0, 1},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
        };

        double[][] saidasEsperadas = {
//              {gripe, dengue}
                {1, 0},
                {0, 1},
                {1, 0},
                {0, 1}
        };

        // Treinamento da rede neural
        mlp.treinar(entradas, saidasEsperadas, epocas);

        // Teste da rede neural com novos dados
        double[] entradaTeste = {1, 0, 1, 1};
        double[] resultado = mlp.feedForward(entradaTeste);

        // Exibição do resultado
        System.out.println("Resultado da rede MLP para a entrada de teste:");
        for (double valor : resultado) {
            System.out.println(valor);
        }
    }
}