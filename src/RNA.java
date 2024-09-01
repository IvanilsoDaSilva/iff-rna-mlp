import ativacaostrategy.AtivacaoStrategy;
import java.util.Random;

/**
 * Representa uma Rede Neural Artificial com camadas ocultas.
 */
public class RNA {
    Random rand = new Random();
    private int neuroniosEntrada;
    private int neuroniosOcultos;
    private int neuroniosSaida;
    private double[][] pesosOcultos;
    private double[][] pesosSaida;
    private double taxaAprendizado;
    private AtivacaoStrategy ativacao;
    private boolean mostrarPassos;
    private boolean biasAleatorio;

    /**
     * Construtor para inicializar uma rede neural MLP com os parâmetros especificados.
     *
     * @param neuroniosEntrada   Número de neurônios na camada de entrada.
     * @param neuroniosOcultos   Número de neurônios na camada oculta.
     * @param neuroniosSaida     Número de neurônios na camada de saída.
     * @param taxaAprendizado    Taxa de aprendizado usada para atualizar os pesos durante o treinamento.
     * @param ativacao           Função de ativação a ser usada nas camadas da rede.
     * @param mostrarPassos      Indica se deve mostrar informações detalhadas sobre a estrutura da rede.
     */
    public RNA(
            int neuroniosEntrada, int neuroniosOcultos, int neuroniosSaida, double taxaAprendizado,
            AtivacaoStrategy ativacao, boolean mostrarPassos, boolean biasAleatorio) {
        this.neuroniosEntrada = neuroniosEntrada;
        this.neuroniosOcultos = neuroniosOcultos;
        this.neuroniosSaida = neuroniosSaida;
        this.taxaAprendizado = taxaAprendizado;
        this.ativacao = ativacao;
        this.mostrarPassos = mostrarPassos;
        this.biasAleatorio = biasAleatorio;

        this.desenharRede();
        this.iniciarSinapses();
    }

    /**
     * Desenha a estrutura da rede neural e mostra informações detalhadas sobre os pesos e neurônios,
     * se a opção 'mostrarPassos' estiver habilitada.
     */
    private void desenharRede() {
        if (this.mostrarPassos) {
            System.out.println("--------------Desenho do RNA");
            System.out.println("Neuronios de entrada com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosEntrada; i++) {
                System.out.printf("[x%d],", i);
            }
            System.out.println("\nSinapses dos Neuronios ocultos com o bias: ");
            for (int i = 0; i < this.neuroniosEntrada + 1; i++) {
                for (int j = 0; j < this.neuroniosOcultos; j++) {
//                    System.out.printf("[w%d%d=%.2f],", i, j, pesosSaida[i][j]);
                    System.out.printf("[w%d%d],", i, j);
                }
            }
            System.out.println("\nNeuronios ocultos com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosOcultos; i++) {
                System.out.printf("[x%d],", i);
            }
            System.out.println("\nSinapses dos neuronios de saida com o bias: ");
            for (int i = 0; i < this.neuroniosOcultos + 1; i++) {
                for (int j = 0; j < this.neuroniosSaida; j++) {
//                    System.out.printf("[w%d%d=%.2f],", i, j, pesosSaida[i][j]);
                    System.out.printf("[w%d%d],", i, j);
                }
            }
            System.out.println("\nNeuronios de saida com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosSaida; i++) {
                System.out.printf("[x%d],", i);
            }
        }
    }

    private void iniciarSinapses(){
        pesosOcultos = new double[this.neuroniosEntrada + 1][this.neuroniosOcultos];
        pesosSaida = new double[this.neuroniosOcultos + 1][this.neuroniosSaida];

        for (int i = 0; i < this.neuroniosEntrada + 1; i++) {
            for (int j = 0; j < this.neuroniosOcultos; j++) {
                pesosOcultos[i][j] = rand.nextDouble() - 0.5;
                if (i == 0) {
                    pesosOcultos[i][j] = 1;
                    if (biasAleatorio)
                        pesosOcultos[i][j] = rand.nextDouble() - 0.5; // Inicialização do bias aleatorio
                }
            }
        }

        for (int i = 0; i < this.neuroniosOcultos + 1; i++) {
            for (int j = 0; j < this.neuroniosSaida; j++) {
                pesosSaida[i][j] = rand.nextDouble() - 0.5;
                if (i == 0) {
                    pesosSaida[i][j] = 1;
                    if (biasAleatorio)
                        pesosOcultos[i][j] = rand.nextDouble() - 0.5; // Inicialização do bias aleatorio
                }
            }
        }


    }

    /**
     * Treina a rede neural usando o algoritmo de retropropagação com o conjunto de dados fornecido.
     *
     * @param entradas   Matrizes de entradas de treinamento, onde cada linha representa um exemplo.
     * @param saidas     Matrizes de saídas esperadas para cada exemplo de treinamento.
     * @param epocas     Número de épocas para o treinamento.
     */
    public void treinar(double[][] entradas, double[][] saidas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (int i = 0; i < entradas.length; i++) {
                // Passo para frente (Forward pass)
                double[] saidaCamadaOculta = new double[this.neuroniosOcultos];
                double[] saidaCamadaSaida = new double[this.neuroniosSaida];

                // Calcular a saída da camada oculta
                for (int j = 0; j < this.neuroniosOcultos; j++) {
                    double ativacao = pesosOcultos[this.neuroniosEntrada][j]; // Bias da camada oculta
                    for (int k = 0; k < this.neuroniosEntrada; k++) {
                        ativacao += entradas[i][k] * pesosOcultos[k][j];
                    }
                    saidaCamadaOculta[j] = ativacao; // Aplicar a função de ativação
                    saidaCamadaOculta[j] = this.ativacao.ativar(saidaCamadaOculta[j]);
                }

                // Calcular a saída da camada de saída
                for (int j = 0; j < this.neuroniosSaida; j++) {
                    double ativacao = pesosSaida[this.neuroniosOcultos][j]; // Bias da camada de saída
                    for (int k = 0; k < this.neuroniosOcultos; k++) {
                        ativacao += saidaCamadaOculta[k] * pesosSaida[k][j];
                    }
                    saidaCamadaSaida[j] = ativacao; // Aplicar a função de ativação
                    saidaCamadaSaida[j] = this.ativacao.ativar(saidaCamadaSaida[j]);
                }

                // Retropropagação para ajustar os pesos da camada de saída
                double[] erroCamadaSaida = new double[this.neuroniosSaida];
                for (int j = 0; j < this.neuroniosSaida; j++) {
                    erroCamadaSaida[j] = saidas[i][j] - saidaCamadaSaida[j];
                    for (int k = 0; k < this.neuroniosOcultos; k++) {
                        double derivadaAtivacao = this.ativacao.derivada(saidaCamadaSaida[j]);
                        pesosSaida[k][j] += this.taxaAprendizado * erroCamadaSaida[j] * derivadaAtivacao * saidaCamadaOculta[k];
                    }
                    pesosSaida[this.neuroniosOcultos][j] += this.taxaAprendizado * erroCamadaSaida[j] * this.ativacao.derivada(saidaCamadaSaida[j]);
                }

                // Retropropagação para ajustar os pesos da camada oculta
                double[] erroCamadaOculta = new double[this.neuroniosOcultos];
                for (int j = 0; j < this.neuroniosOcultos; j++) {
                    erroCamadaOculta[j] = 0.0;
                    for (int k = 0; k < this.neuroniosSaida; k++) {
                        erroCamadaOculta[j] += erroCamadaSaida[k] * pesosSaida[j][k];
                    }
                    erroCamadaOculta[j] *= this.ativacao.derivada(saidaCamadaOculta[j]);
                    for (int k = 0; k < this.neuroniosEntrada; k++) {
                        pesosOcultos[k][j] += this.taxaAprendizado * erroCamadaOculta[j] * entradas[i][k];
                    }
                    pesosOcultos[this.neuroniosEntrada][j] += this.taxaAprendizado * erroCamadaOculta[j];
                }
            }
        }
    }

    /**
     * Faz uma previsão com base na entrada fornecida usando a rede neural treinada.
     *
     * @param entrada   Vetor de entrada para a qual a previsão deve ser feita.
     * @return          Vetor contendo a saída da rede neural para a entrada fornecida.
     */
    public double[] prever(double[] entrada) {
        double[] saidaCamadaOculta = new double[this.neuroniosOcultos];
        double[] saidaCamadaSaida = new double[this.neuroniosSaida];

        // Passo para frente na camada oculta
        for (int j = 0; j < this.neuroniosOcultos; j++) {
            double ativacao = pesosOcultos[this.neuroniosEntrada][j]; // Bias da camada oculta
            for (int k = 0; k < this.neuroniosEntrada; k++) {
                ativacao += entrada[k] * pesosOcultos[k][j];
            }
            saidaCamadaOculta[j] = ativacao; // Aplicar a função de ativação
            saidaCamadaOculta[j] = this.ativacao.ativar(saidaCamadaOculta[j]);
        }

        // Passo para frente na camada de saída
        for (int j = 0; j < this.neuroniosSaida; j++) {
            double ativacao = pesosSaida[this.neuroniosOcultos][j]; // Bias da camada de saída
            for (int k = 0; k < this.neuroniosOcultos; k++) {
                ativacao += saidaCamadaOculta[k] * pesosSaida[k][j];
            }

            saidaCamadaSaida[j] = ativacao; // Aplicar a função de ativação
            saidaCamadaSaida[j] = this.ativacao.ativar(saidaCamadaSaida[j]);
        }

        return saidaCamadaSaida;
    }
}
