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

        MLP mlp = new MLP(4, 2, 2, 0.5);
        mlp.treinar(entradas, saidas, 10000);

        // Testar a rede com uma nova entrada
        double[] entradaTeste = {1, 0, 1, 0}; // Deve resultar em gripe
        double[] resultado = mlp.prever(entradaTeste);

        System.out.println("Gripe: " + resultado[0]);
        System.out.println("Dengue: " + resultado[1]);
    }
}