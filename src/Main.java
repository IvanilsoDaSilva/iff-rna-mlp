import ativacaostrategy.AtivacaoBinariaStrategy;
import ativacaostrategy.AtivacaoSigmoideStrategy;

public class Main {
    public static void main(String[] args) {
        // Conjunto de treinamento
        double[][] entradas = {
        //	{Tosse, Mancha na pele, Corisa, Baixa plaqueta}
            {1, 0, 1, 0},
            {0, 1, 0, 1},
            {1, 0, 0, 1},
            {0, 1, 1, 0}
        };

        double[][] saidas = {
        //	{Gripe, Dengue}
            {1, 0},
            {0, 1},
            {0, 1},
            {1, 0}
        };

        // Instancia e treino da rede
        RNA mlp = new RNA(
                4, 2, 2, 0.5,
                new AtivacaoSigmoideStrategy(), true, false
                );
        mlp.treinar(entradas, saidas, 1);

        // Testar a rede com novas entradas
//        double[] entradaTeste = {1, 0, 1, 0}; //gripe
        double[] entradaTeste = {0, 1, 0, 1}; //dengue

        double[] resultado = mlp.prever(entradaTeste);
        System.out.println();
        System.out.println("Gripe: " + resultado[0]);
        System.out.println("Dengue: " + resultado[1]);
    }
}