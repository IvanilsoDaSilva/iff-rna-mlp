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
    public RNA(int neuroniosEntrada, int neuroniosOcultos, int neuroniosSaida, double taxaAprendizado, AtivacaoStrategy ativacao, boolean mostrarPassos, boolean biasAleatorio) {
        this.neuroniosEntrada = neuroniosEntrada;
        this.neuroniosOcultos = neuroniosOcultos;
        this.neuroniosSaida = neuroniosSaida;
        this.taxaAprendizado = taxaAprendizado;
        this.ativacao = ativacao;
        this.mostrarPassos = mostrarPassos;
        this.biasAleatorio = biasAleatorio;

        System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                "0. Desenho do RNA");
        this.desenharRNA();
        System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                "1. Iniciar sinapse");
        this.iniciarSinapses();
        this.desenharSinapses();
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

    public void treinar(double[][] entradas, double[][] saidas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (int i = 0; i < entradas.length; i++) {
                // Passo para frente (Forward pass)
                System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                        "2."+i+" Calcular Saida da camada de entrada");
                double[] saidaCamadaOculta = calcularSaida(entradas[i], pesosOcultos, this.neuroniosOcultos, this.neuroniosEntrada);
                System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                        "3."+i+" Calcular Saida da camada de saida");
                double[] saidaCamadaSaida = calcularSaida(saidaCamadaOculta, pesosSaida, this.neuroniosSaida, this.neuroniosOcultos);
                System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                        "4."+i+" Calculo do erro da camada de saída");
                double[] erroCamadaSaida = calcularErro(saidas[i], saidaCamadaSaida, null, null);
                System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                        "5."+i+" Atualização dos pesos da camada de saída");
                atualizarPesos(pesosSaida, erroCamadaSaida, saidaCamadaOculta, this.neuroniosOcultos);
                System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                        "6."+i+" Cálculo do erro da camada oculta");
                double[] erroCamadaOculta = calcularErro(null, saidaCamadaOculta, erroCamadaSaida, pesosSaida);
                System.out.println("------------------------------------------------------------------------------------------------------------------ " +
                        "7."+i+" Atualização dos pesos da camada oculta");
                atualizarPesos(pesosOcultos, erroCamadaOculta, entradas[i], this.neuroniosEntrada);
            }
        }
    }
    /**
     * Calcula a saída de uma camada da rede neural (oculta ou de saída).
     *
     * @param entradas         Entradas para a camada atual.
     * @param pesos            Pesos da camada atual.
     * @param neuroniosDestino Número de neurônios na camada de destino.
     * @param neuroniosOrigem  Número de neurônios na camada de origem.
     * @return                 Saída calculada da camada.
     */
    private double[] calcularSaida(double[] entradas, double[][] pesos, int neuroniosDestino, int neuroniosOrigem) {
        // v = soma(xi*wi)
        double[] y = new double[neuroniosDestino];
        for (int i = 0; i < neuroniosDestino; i++) {
            double v = pesos[neuroniosOrigem][i]; // Bias
            for (int j = 0; j < neuroniosOrigem; j++) {
                v += entradas[j] * pesos[j][i];
            }
            //
            y[i] = this.ativacao.ativar(v);
        }
        this.desenharCalcularSaida(entradas, pesos, neuroniosDestino, neuroniosOrigem, y);
        return y;
    }
    /**
     * Calcula o erro para uma camada da rede neural.
     *
     * @param saidasEsperadas Saídas esperadas para o exemplo atual (somente para camada de saída).
     * @param saidasCalculadas Saídas calculadas pela rede na camada atual.
     * @param errosCamadaSeguinte Erros da camada seguinte (somente para camada oculta).
     * @param pesosCamadaSeguinte Pesos que conectam a camada atual à camada seguinte (somente para camada oculta).
     * @return Vetor de erros da camada atual.
     */
    private double[] calcularErro(double[] saidasEsperadas, double[] saidasCalculadas, double[] errosCamadaSeguinte, double[][] pesosCamadaSeguinte) {
        double[] erro = new double[saidasCalculadas.length];

        // Se saidasEsperadas não for nulo, estamos calculando o erro da camada de saída
        if (saidasEsperadas != null) {
            for (int j = 0; j < saidasCalculadas.length; j++) {
                erro[j] = saidasEsperadas[j] - saidasCalculadas[j];
                erro[j] *= this.ativacao.derivada(saidasCalculadas[j]);
            }
        }
        // Se errosCamadaSeguinte não for nulo, estamos calculando o erro da camada oculta
        else if (errosCamadaSeguinte != null && pesosCamadaSeguinte != null) {
            for (int j = 0; j < erro.length; j++) {
                double soma = 0.0;
                for (int k = 0; k < errosCamadaSeguinte.length; k++) {
                    soma += errosCamadaSeguinte[k] * pesosCamadaSeguinte[j][k];
                }
                erro[j] = soma * this.ativacao.derivada(saidasCalculadas[j]);
            }
        }

        return erro;
    }
    /**
     * Atualiza os pesos de uma camada com base no erro calculado e nas entradas fornecidas.
     *
     * @param pesos           Pesos da camada a serem atualizados.
     * @param erro            Erros calculados para a camada.
     * @param entradas        Entradas para a camada atual.
     * @param neuroniosOrigem Número de neurônios na camada de origem.
     */
    private void atualizarPesos(double[][] pesos, double[] erro, double[] entradas, int neuroniosOrigem) {
        for (int j = 0; j < erro.length; j++) {
            for (int k = 0; k < neuroniosOrigem; k++) {
                pesos[k][j] += this.taxaAprendizado * erro[j] * entradas[k];
            }
            // Atualização do bias
            pesos[neuroniosOrigem][j] += this.taxaAprendizado * erro[j];
        }
    }

    private void desenharRNA() {
        if (this.mostrarPassos) {
            System.out.println("Neuronios de entrada com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosEntrada; i++) {
                System.out.printf("[x%d]", i);
            }
            System.out.println("\nSinapses dos Neuronios ocultos com o bias: ");
            for (int i = 0; i < this.neuroniosEntrada + 1; i++) {
                for (int j = 0; j < this.neuroniosOcultos; j++) {
                    System.out.printf("[w%d%d]", j, i);
                }
            }
            System.out.println("\nNeuronios ocultos com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosOcultos; i++) {
                System.out.printf("[x%d]", i);
            }
            System.out.println("\nSinapses dos neuronios de saida com o bias: ");
            for (int i = 0; i < this.neuroniosOcultos + 1; i++) {
                for (int j = 0; j < this.neuroniosSaida; j++) {
                    System.out.printf("[w%d%d]", j, i);
                }
            }
            System.out.println("\nNeuronios de saida com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosSaida; i++) {
                System.out.printf("[x%d]", i);
            }
            System.out.println();
        }
    }
    private void desenharConjuntoTreinamento() {
        if (this.mostrarPassos) {

        }
    }
    private void desenharSinapses(){
        if (this.mostrarPassos) {
            System.out.print("Camada de entrada:\n");
            for (int i = 0; i < this.neuroniosOcultos; i++) {
                System.out.print("wi="+i+"[");
                for (int j = 0; j < this.neuroniosEntrada+1; j++) {
                    System.out.printf("%6.1f, ", pesosOcultos[j][i]);
                }
                System.out.print("]\n");
            }
            System.out.print("Camada escondida:");
            for (int i = 0; i < this.neuroniosSaida; i++) {
                System.out.print("\nwj="+i+"[");
                for (int j = 0; j < this.neuroniosSaida+1; j++) {
                    System.out.printf("%6.1f, ", pesosSaida[j][i]);
                }
                System.out.print("]");
            }
            System.out.println();
        }
    }
    private void desenharCalcularSaida(double[] entradas, double[][] pesos, int neuroniosDestino, int neuroniosOrigem, double[] saida) {
        if (this.mostrarPassos) {
            // Exibindo a camada de entrada
            System.out.print("Camada de entrada:\n[ ");
            for (int i = 0; i < neuroniosOrigem; i++) {
                System.out.printf("%4.1f ", entradas[i]);
            }
            System.out.println("]");

            // Exibindo a matriz de pesos no sentido contrário
            System.out.println("Matriz de pesos (exibida no sentido contrário):");
            for (int j = 0; j < neuroniosOrigem; j++) {
                System.out.print("[ ");
                for (int i = 0; i < neuroniosDestino; i++) {
                    System.out.printf("%6.1f ", pesos[j][i]);
                }
                System.out.println("]");
            }

            // Visualização da multiplicação
            System.out.println("Visualização da multiplicação:");
            for (int i = 0; i < neuroniosDestino; i++) {
                System.out.print("[ ");
                for (int j = 0; j < neuroniosOrigem; j++) {
                    System.out.printf("%6.1f * %6.1f", entradas[j], pesos[j][i]);
                    if (j < neuroniosOrigem - 1) {
                        System.out.print(" + ");
                    }
                }
                System.out.printf(" ] = %6.1f \n", saida[i]);
            }
        }
    }
}
