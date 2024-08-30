import java.util.Random;

/**
 * Representa uma rede neural MLP (Perceptron Multicamadas) com uma camada oculta.
 */
public class MLP {
    private double[][] pesosEntradaOculta;
    private double[][] pesosOcultaSaida;
    private double taxaAprendizado;

    /**
     * Construtor da rede neural MLP.
     *
     * @param tamanhoEntrada Número de neurônios na camada de entrada.
     * @param tamanhoOculta Número de neurônios na camada oculta.
     * @param tamanhoSaida Número de neurônios na camada de saída.
     * @param taxaAprendizado Taxa de aprendizado para atualizar os pesos.
     */
    public MLP(int tamanhoEntrada, int tamanhoOculta, int tamanhoSaida, double taxaAprendizado) {
        pesosEntradaOculta = new double[tamanhoEntrada][tamanhoOculta];
        pesosOcultaSaida = new double[tamanhoOculta][tamanhoSaida];
        this.taxaAprendizado = taxaAprendizado;

        inicializarPesos(pesosEntradaOculta);
        inicializarPesos(pesosOcultaSaida);
    }

    /**
     * Inicializa os pesos com valores aleatórios pequenos.
     *
     * @param pesos Matriz de pesos a ser inicializada.
     */
    private void inicializarPesos(double[][] pesos) {
        Random random = new Random();
        for (int i = 0; i < pesos.length; i++) {
            for (int j = 0; j < pesos[i].length; j++) {
                pesos[i][j] = random.nextDouble() * 0.1;  // Inicializa pesos com pequenos valores aleatórios
            }
        }
    }

    /**
     * Calcula a função de ativação sigmoide.
     *
     * @param x Valor de entrada para a função sigmoide.
     * @return Resultado da função sigmoide.
     */
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Calcula a derivada da função de ativação sigmoide.
     *
     * @param x Valor de entrada para a função sigmoide.
     * @return Derivada da função sigmoide.
     */
    private double sigmoidDerivada(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    /**
     * Realiza a propagação para frente (feedforward) na rede neural.
     *
     * @param entrada Vetor de entrada para a rede neural.
     * @return Vetor de saída da rede neural.
     */
    public double[] feedForward(double[] entrada) {
        double[] camadaOculta = new double[pesosEntradaOculta[0].length];
        double[] camadaSaida = new double[pesosOcultaSaida[0].length];

        // Calcula a saída da camada oculta
        for (int j = 0; j < camadaOculta.length; j++) {
            camadaOculta[j] = 0;
            for (int i = 0; i < entrada.length; i++) {
                camadaOculta[j] += entrada[i] * pesosEntradaOculta[i][j];
            }
            camadaOculta[j] = sigmoid(camadaOculta[j]);
        }

        // Calcula a saída da camada de saída
        for (int k = 0; k < camadaSaida.length; k++) {
            camadaSaida[k] = 0;
            for (int j = 0; j < camadaOculta.length; j++) {
                camadaSaida[k] += camadaOculta[j] * pesosOcultaSaida[j][k];
            }
            camadaSaida[k] = sigmoid(camadaSaida[k]);
        }

        return camadaSaida;
    }

    /**
     * Realiza a retropropagação (backpropagation) para atualizar os pesos da rede neural.
     *
     * @param entrada Vetor de entrada para a rede neural.
     * @param esperado Vetor com os valores esperados de saída.
     * @param camadaOculta Vetor com os valores da camada oculta.
     * @param camadaSaida Vetor com os valores da camada de saída.
     */
    private void backpropagation(double[] entrada, double[] esperado, double[] camadaOculta, double[] camadaSaida) {
        // Cálculo do erro na camada de saída
        double[] errosSaida = new double[camadaSaida.length];
        double[] gradientesSaida = new double[camadaSaida.length];
        for (int k = 0; k < camadaSaida.length; k++) {
            errosSaida[k] = esperado[k] - camadaSaida[k];
            gradientesSaida[k] = errosSaida[k] * sigmoidDerivada(camadaSaida[k]);
        }

        // Cálculo do erro na camada oculta
        double[] errosOculta = new double[camadaOculta.length];
        double[] gradientesOculta = new double[camadaOculta.length];
        for (int j = 0; j < camadaOculta.length; j++) {
            errosOculta[j] = 0;
            for (int k = 0; k < camadaSaida.length; k++) {
                errosOculta[j] += gradientesSaida[k] * pesosOcultaSaida[j][k];
            }
            gradientesOculta[j] = errosOculta[j] * sigmoidDerivada(camadaOculta[j]);
        }

        // Atualização dos pesos para a camada de saída
        for (int j = 0; j < camadaOculta.length; j++) {
            for (int k = 0; k < camadaSaida.length; k++) {
                pesosOcultaSaida[j][k] += taxaAprendizado * gradientesSaida[k] * camadaOculta[j];
            }
        }

        // Atualização dos pesos para a camada oculta
        for (int i = 0; i < entrada.length; i++) {
            for (int j = 0; j < camadaOculta.length; j++) {
                pesosEntradaOculta[i][j] += taxaAprendizado * gradientesOculta[j] * entrada[i];
            }
        }
    }

    /**
     * Treina a rede neural com um conjunto de dados de entrada e saída esperada.
     *
     * @param entradas Matriz de dados de entrada para treinamento.
     * @param saidasEsperadas Matriz de dados de saída esperada.
     * @param epocas Número de épocas para o treinamento.
     */
    public void treinar(double[][] entradas, double[][] saidasEsperadas, int epocas) {
        for (int epoch = 0; epoch < epocas; epoch++) {
            for (int i = 0; i < entradas.length; i++) {
                double[] entrada = entradas[i];
                double[] esperado = saidasEsperadas[i];

                // Passo forward
                double[] camadaOculta = new double[pesosEntradaOculta[0].length];
                double[] camadaSaida = feedForward(entrada);

                // Retropropagação para atualizar os pesos
                backpropagation(entrada, esperado, camadaOculta, camadaSaida);
            }
        }
    }
}