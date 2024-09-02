import ativacaostrategy.AtivacaoStrategy;
import java.util.Random;

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

    public RNA(int neuroniosEntrada, int neuroniosOcultos, int neuroniosSaida, double taxaAprendizado, AtivacaoStrategy ativacao, boolean mostrarPassos, boolean biasAleatorio) {
        this.neuroniosEntrada = neuroniosEntrada;
        this.neuroniosOcultos = neuroniosOcultos;
        this.neuroniosSaida = neuroniosSaida;
        this.taxaAprendizado = taxaAprendizado;
        this.ativacao = ativacao;
        this.mostrarPassos = mostrarPassos;
        this.biasAleatorio = biasAleatorio;

        System.out.println("----------------------------------------------------------------------------------- " +
                "0. Desenho do RNA");
        this.desenharRNA();
        System.out.println("----------------------------------------------------------------------------------- " +
                "1. Iniciar sinapse");
        this.iniciarSinapses();
        this.desenharSinapses();
    }

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
                System.out.println("----------------------------------------------------------------------------------- " +
                        "2."+i+" Multiplicar entrada por pesos na camada oculta (xi*wi)");
                double[] valorEntradaVezesSinapseOculta = multiplicarPesoEntrada(entradas[i], this.pesosOcultos, this.neuroniosOcultos, this.neuroniosEntrada);
                this.desenharMultiplicarPesoEntrada(entradas[i], this.pesosOcultos, this.neuroniosOcultos, this.neuroniosEntrada, valorEntradaVezesSinapseOculta);
                System.out.println("----------------------------------------------------------------------------------- " +
                        "3."+i+" Função ativação na camada oculta ("+this.ativacao.getExpressao()+")");
                double[] saidaCamadaOculta = aplicarAtivacao(valorEntradaVezesSinapseOculta);
                this.desenharAplicarAtivacao(valorEntradaVezesSinapseOculta, saidaCamadaOculta);
                System.out.println("----------------------------------------------------------------------------------- " +
                        "4."+i+" Multiplicar entrada por pesos na camada de saída (yi*wi)");
                double[] valorEntradaVezesSinapseSaida = multiplicarPesoEntrada(saidaCamadaOculta, this.pesosSaida, this.neuroniosSaida, this.neuroniosOcultos);
                this.desenharMultiplicarPesoEntrada(saidaCamadaOculta, this.pesosSaida, this.neuroniosSaida, this.neuroniosOcultos, valorEntradaVezesSinapseSaida);
                System.out.println("----------------------------------------------------------------------------------- " +
                        "5."+i+" Função ativação na camada de saída ("+this.ativacao.getExpressao()+")");
                double[] saidaCamadaSaida = aplicarAtivacao(valorEntradaVezesSinapseSaida);
                this.desenharAplicarAtivacao(valorEntradaVezesSinapseSaida, saidaCamadaSaida);
                System.out.println("----------------------------------------------------------------------------------- " +
                        "6."+i+" Cálculo do erro e atualização dos pesos");
                double[] erroCamadaSaida = calcularErro(saidas[i], saidaCamadaSaida, null, null);
                desenharCalculoErro(saidas[i], saidaCamadaSaida, erroCamadaSaida, null, "Saída");
                atualizarPesos(pesosSaida, erroCamadaSaida, saidaCamadaOculta, this.neuroniosOcultos);
                double[] erroCamadaOculta = calcularErro(null, saidaCamadaOculta, erroCamadaSaida, pesosSaida);
                desenharCalculoErro(null, saidaCamadaOculta, erroCamadaOculta, pesosSaida, "Oculta");
                atualizarPesos(pesosOcultos, erroCamadaOculta, entradas[i], this.neuroniosEntrada);
                desenharSinapses();
            }
        }
    }

    private double[] multiplicarPesoEntrada(double[] entradas, double[][] pesos, int neuroniosDestino, int neuroniosOrigem) {
        double[] somas = new double[neuroniosDestino];
        for (int i = 0; i < neuroniosDestino; i++) {
            double soma = pesos[neuroniosOrigem][i]; // Bias
            for (int j = 0; j < neuroniosOrigem; j++) {
                soma += entradas[j] * pesos[j][i];
            }
            somas[i] = soma;
        }
        return somas;
    }

    // Função para aplicar a função de ativação
    private double[] aplicarAtivacao(double[] somas) {
        double[] ativacoes = new double[somas.length];
        for (int i = 0; i < somas.length; i++) {
            ativacoes[i] = this.ativacao.ativar(somas[i]);
        }
        return ativacoes;
    }

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
            System.out.println("----------------------------------------------------------------------------------- Neuronios de entrada com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosEntrada; i++) {
                System.out.printf("[x%d]", i);
            }
            System.out.println("\n----------------------------------------------------------------------------------- Sinapses dos Neuronios ocultos com o bias: ");
            for (int i = 0; i < this.neuroniosEntrada + 1; i++) {
                for (int j = 0; j < this.neuroniosOcultos; j++) {
                    System.out.printf("[w%d%d]", j, i);
                }
            }
            System.out.println("\n----------------------------------------------------------------------------------- Neuronios ocultos com o bias: ");
            System.out.print("[b]");
            for (int i = 0; i < this.neuroniosOcultos; i++) {
                System.out.printf("[x%d]", i);
            }
            System.out.println("\n----------------------------------------------------------------------------------- Sinapses dos neuronios de saida com o bias: ");
            for (int i = 0; i < this.neuroniosOcultos + 1; i++) {
                for (int j = 0; j < this.neuroniosSaida; j++) {
                    System.out.printf("[w%d%d]", j, i);
                }
            }
            System.out.println("\n----------------------------------------------------------------------------------- Neuronios de saida com o bias: ");
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
            System.out.print("----------------------------------------------------------------------------------- Camada de entrada:\n");
            for (int i = 0; i < this.neuroniosOcultos; i++) {
                System.out.print("wi="+i+"[");
                for (int j = 0; j < this.neuroniosEntrada+1; j++) {
                    System.out.printf("%6.1f, ", pesosOcultos[j][i]);
                }
                System.out.print("]\n");
            }
            System.out.print("----------------------------------------------------------------------------------- Camada escondida:");
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
    private void desenharMultiplicarPesoEntrada(double[] entradas, double[][] pesos, int neuroniosDestino, int neuroniosOrigem, double[] saida) {
        if (this.mostrarPassos) {
            // Exibindo a camada de entrada
            System.out.print("----------------------------------------------------------------------------------- Camada de entrada:\n[ ");
            for (int i = 0; i < neuroniosOrigem; i++) {
                System.out.printf("%4.1f ", entradas[i]);
            }
            System.out.println("]");

            // Exibindo a matriz de pesos
            System.out.println("----------------------------------------------------------------------------------- Matriz de pesos:");
            for (int j = 0; j < neuroniosOrigem; j++) {
                System.out.print("[ ");
                for (int i = 0; i < neuroniosDestino; i++) {
                    System.out.printf("%6.1f ", pesos[j][i]);
                }
                System.out.println("]");
            }

            // Visualização da multiplicação
            System.out.println("----------------------------------------------------------------------------------- Visualização da multiplicação:");
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
    private void desenharAplicarAtivacao(double[] v, double[] y) {
        if (this.mostrarPassos) {
            for (int i = 0; i < v.length; i++) {
                System.out.printf("v%d = %6.1f -> y%d = %6.1f\n", i, v[i], i, y[i]);
            }
        }
    }
    private void desenharCalculoErro(double[] saidasEsperadas, double[] saidasCalculadas, double[] erro, double[][] pesosCamadaSeguinte, String camada) {
        if (this.mostrarPassos) {
            System.out.println("----------------------------------------------------------------------------------- " +
                    "Cálculo do erro da camada " + camada);

            if (saidasEsperadas != null) { // Erro da camada de saída
                for (int i = 0; i < erro.length; i++) {
                    System.out.printf("Erro[%d] = (Saída Esperada[%d] - Saída Calculada[%d]) * Derivada(Saída Calculada[%d])\n", i, i, i, i);
                    System.out.printf("Erro[%d] = (%.4f - %.4f) * Derivada(%.4f)\n",
                            i, saidasEsperadas[i], saidasCalculadas[i], saidasCalculadas[i]);
                    System.out.printf("Erro[%d] = %.4f\n\n", i, erro[i]);
                }
            } else if (pesosCamadaSeguinte != null) { // Erro da camada oculta
                for (int i = 0; i < erro.length; i++) {
                    System.out.printf("Erro[%d] = Somatório(Erro da Camada Seguinte * Peso) * Derivada(Saída Calculada[%d])\n", i, i);
                    double soma = 0.0;
                    for (int j = 0; j < pesosCamadaSeguinte[i].length; j++) {
                        System.out.printf("Peso[%d][%d] * Erro da Camada Seguinte[%d] + ", i, j, j);
                        soma += pesosCamadaSeguinte[i][j] * erro[j];
                    }
                    System.out.printf("\b\b\b = %.4f\n", soma);
                    System.out.printf("Erro[%d] = %.4f * Derivada(%.4f)\n", i, soma, saidasCalculadas[i]);
                    System.out.printf("Erro[%d] = %.4f\n\n", i, erro[i]);
                }
            }
        }
    }

}
