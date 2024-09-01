import java.util.Random;

/**
 * Classe que implementa uma rede neural Perceptron Multicamadas (MLP).
 * Esta classe define a estrutura e os métodos necessários para criar, treinar 
 * e usar uma rede neural MLP com uma camada de entrada, uma camada oculta e uma camada de saída.
 */
public class MLP {
    Random rand = new Random();
    // Parâmetros da rede
    private int neuroniosEntrada;   // Número de neurônios na camada de entrada
    private int neuroniosOcultos;   // Número de neurônios na camada oculta
    private int neuroniosSaida;     // Número de neurônios na camada de saída
    private double taxaAprendizado; // Taxa de aprendizado para ajuste dos pesos

    // Pesos e bias
    private double[][] pesosOcultos;    // Pesos entre a camada de entrada e a camada oculta
    private double[][] pesosSaida;      // Pesos entre a camada oculta e a camada de saída
    private double[] biasOculto;        // Bias da camada oculta
    private double[] biasSaida;         // Bias da camada de saída

    /**
     * Construtor da classe MLP.
     * Este construtor inicializa uma rede neural perceptron multicamadas (MLP) com uma 
     * camada de entrada, uma camada oculta e uma camada de saída. 
     * Os parâmetros de configuração da rede, como o número de neurônios em cada camada 
     * e a taxa de aprendizado, são passados como argumentos. Os pesos e biases são 
     * inicializados com valores aleatórios para permitir que a rede comece a aprender.
     * 
     * @param neuroniosEntrada     Número de neurônios na camada de entrada. Define o tamanho
     *                             do vetor de entrada da rede.
     * @param neuroniosOcultos     Número de neurônios na camada oculta. Controla a capacidade
     *                             da rede de capturar padrões complexos nos dados.
     * @param neuroniosSaida       Número de neurônios na camada de saída. Define o tamanho do
     *                             vetor de saída da rede, que pode representar classes de classificação 
     *                             ou valores contínuos.
     * @param taxaAprendizado      Taxa de aprendizado usada durante o treinamento da rede. 
     *                             Controla a velocidade com que a rede ajusta seus pesos durante 
     *                             o processo de retropropagação.
     */
    public MLP(int neuroniosEntrada, int neuroniosOcultos, int neuroniosSaida, double taxaAprendizado) {
        this.neuroniosEntrada = neuroniosEntrada;
        this.neuroniosOcultos = neuroniosOcultos;
        this.neuroniosSaida = neuroniosSaida;
        this.taxaAprendizado = taxaAprendizado;
        
        // Inicializar os pesos e bias com valores aleatórios
        pesosOcultos = new double[this.neuroniosEntrada][this.neuroniosOcultos];
        pesosSaida = new double[this.neuroniosOcultos][this.neuroniosSaida];
        biasOculto = new double[this.neuroniosOcultos];
        biasSaida = new double[this.neuroniosSaida];

        // Inicialização dos pesos da camada oculta
        for (int i = 0; i < this.neuroniosEntrada; i++) {
            for (int j = 0; j < this.neuroniosOcultos; j++) {
                pesosOcultos[i][j] = rand.nextDouble() - 0.5;
            }
        }

        // Inicialização dos bias da camada oculta
        for (int j = 0; j < this.neuroniosOcultos; j++) {
//          biasOculto[j] = rand.nextDouble() - 0.5; // Inicialização do bias aleatorio
            biasOculto[j] = 1; // Inicialização do bias como 1

            // Inicialização dos pesos da camada de saída
            for (int k = 0; k < this.neuroniosSaida; k++) {
                pesosSaida[j][k] = rand.nextDouble() - 0.5;
            }
        }

        // Inicialização dos bias da camada de saída
        for (int k = 0; k < this.neuroniosSaida; k++) {
//          biasSaida[k] = rand.nextDouble() - 0.5; // Inicialização do bias aleatorio
            biasSaida[k] = 1; // Inicialização do bias como 1
        }
    }

    /**
     * Função de ativação sigmoid.
     * A função sigmoid é usada para introduzir não-linearidade na rede neural,
     * mapeando a entrada para um valor entre 0 e 1.
     * 
     * @param x Valor de entrada.
     * @return Resultado da função sigmoid.
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Derivada da função de ativação sigmoid.
     * Esta função calcula a derivada da função sigmoid, que é usada 
     * durante o processo de retropropagação para ajustar os pesos da rede.
     * 
     * @param x Valor de entrada.
     * @return Derivada da função sigmoid.
     */
    private double derivadaSigmoid(double x) {
        return x * (1.0 - x);
    }

    /**
     * Treina a rede neural utilizando o algoritmo de retropropagação.
     * O treinamento envolve múltiplas iterações (épocas), onde a rede ajusta seus pesos
     * para minimizar o erro entre as saídas previstas e as saídas esperadas.
     * 
     * @param entradas Conjunto de entradas para treinamento. Cada linha representa um exemplo de entrada.
     * @param saidas Conjunto de saídas esperadas correspondentes ao conjunto de entradas.
     * @param epocas Número de épocas para o treinamento. Uma época é uma passagem completa pelo conjunto de dados.
     */
    public void treinar(double[][] entradas, double[][] saidas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (int i = 0; i < entradas.length; i++) {
                // Passo para frente (Forward pass)
                double[] saidaCamadaOculta = new double[this.neuroniosOcultos];
                double[] saidaCamadaSaida = new double[this.neuroniosSaida];

                // Calcular a saída da camada oculta
                for (int j = 0; j < this.neuroniosOcultos; j++) {
                    double ativacao = biasOculto[j];
                    for (int k = 0; k < this.neuroniosEntrada; k++) {
                        ativacao += entradas[i][k] * pesosOcultos[k][j];
                    }
                    saidaCamadaOculta[j] = sigmoid(ativacao);
                }

                // Calcular a saída da camada de saída
                for (int j = 0; j < this.neuroniosSaida; j++) {
                    double ativacao = biasSaida[j];
                    for (int k = 0; k < this.neuroniosOcultos; k++) {
                        ativacao += saidaCamadaOculta[k] * pesosSaida[k][j];
                    }
                    saidaCamadaSaida[j] = sigmoid(ativacao);
                }

                // Retropropagação para ajustar os pesos da camada de saída
                double[] erroCamadaSaida = new double[this.neuroniosSaida];
                for (int j = 0; j < this.neuroniosSaida; j++) {
                    erroCamadaSaida[j] = saidas[i][j] - saidaCamadaSaida[j];
                    for (int k = 0; k < this.neuroniosOcultos; k++) {
                        pesosSaida[k][j] += this.taxaAprendizado * erroCamadaSaida[j] * derivadaSigmoid(saidaCamadaSaida[j]) * saidaCamadaOculta[k];
                    }
                    biasSaida[j] += this.taxaAprendizado * erroCamadaSaida[j] * derivadaSigmoid(saidaCamadaSaida[j]);
                }

                // Retropropagação para ajustar os pesos da camada oculta
                double[] erroCamadaOculta = new double[this.neuroniosOcultos];
                for (int j = 0; j < this.neuroniosOcultos; j++) {
                    erroCamadaOculta[j] = 0.0;
                    for (int k = 0; k < this.neuroniosSaida; k++) {
                        erroCamadaOculta[j] += erroCamadaSaida[k] * pesosSaida[j][k];
                    }
                    erroCamadaOculta[j] *= derivadaSigmoid(saidaCamadaOculta[j]);
                    for (int k = 0; k < this.neuroniosEntrada; k++) {
                        pesosOcultos[k][j] += this.taxaAprendizado * erroCamadaOculta[j] * entradas[i][k];
                    }
                    biasOculto[j] += this.taxaAprendizado * erroCamadaOculta[j];
                }
            }
        }
    }

    /**
     * Usa o modelo treinado para prever a saída com base em novas entradas.
     * A função prever utiliza o modelo já treinado para calcular as saídas
     * correspondentes a uma nova entrada.
     * 
     * @param entrada Vetor de entrada com os valores a serem usados como entrada no modelo.
     * @return Vetor de saída com as previsões feitas pela rede neural.
     */
    public double[] prever(double[] entrada) {
        double[] saidaCamadaOculta = new double[this.neuroniosOcultos];
        double[] saidaCamadaSaida = new double[this.neuroniosSaida];

        // Passo para frente na camada oculta
        for (int j = 0; j < this.neuroniosOcultos; j++) {
            double ativacao = biasOculto[j];
            for (int k = 0; k < this.neuroniosEntrada; k++) {
                ativacao += entrada[k] * pesosOcultos[k][j];
            }
            saidaCamadaOculta[j] = sigmoid(ativacao);
        }

        // Passo para frente na camada de saída
        for (int j = 0; j < this.neuroniosSaida; j++) {
            double ativacao = biasSaida[j];
            for (int k = 0; k < this.neuroniosOcultos; k++) {
                ativacao += saidaCamadaOculta[k] * pesosSaida[k][j];
            }
            saidaCamadaSaida[j] = sigmoid(ativacao);
        }

        return saidaCamadaSaida;
    }
}
