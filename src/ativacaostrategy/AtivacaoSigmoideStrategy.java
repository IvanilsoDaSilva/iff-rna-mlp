package ativacaostrategy;

public class AtivacaoSigmoideStrategy implements AtivacaoStrategy {

    @Override
    public double ativar(double x) {return 1.0 / (1.0 + Math.exp(-x));}

    @Override
    public double derivada(double x) {
        return x * (1.0 - x);
    }
}
