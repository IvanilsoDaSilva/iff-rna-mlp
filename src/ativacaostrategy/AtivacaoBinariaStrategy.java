package ativacaostrategy;

public class AtivacaoBinariaStrategy implements AtivacaoStrategy {

    @Override
    public double ativar(double x) {
        return x >= 0 ? 1.0 : 0.0;
    }

    @Override
    public double derivada(double x) {
        // A derivada da função de Heaviside não é bem definida em 0,
        // mas pode ser 0 fora do ponto de descontinuidade.
        return 1.0;
    }
}
