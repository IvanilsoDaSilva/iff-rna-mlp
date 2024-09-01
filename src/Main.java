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

        // Testar a rede com novas entradas
        double[] entradaTeste1 = {1, 0, 1, 0};
        double[] resultado1 = mlp.prever(entradaTeste1);
        System.out.println();
        System.out.println("Entrada: {1, 0, 1, 0}");
        System.out.println("Gripe: " + resultado1[0]);
        System.out.println("Dengue: " + resultado1[1]);

        double[] entradaTeste2 = {0, 1, 0, 1};
        double[] resultado2 = mlp.prever(entradaTeste2);
        System.out.println();
        System.out.println("Entrada: {0, 1, 0, 1}");
        System.out.println("Gripe: " + resultado2[0]);
        System.out.println("Dengue: " + resultado2[1]);

        double[] entradaTeste3 = {1, 0, 0, 1};
        double[] resultado3 = mlp.prever(entradaTeste3);
        System.out.println();
        System.out.println("Entrada: {1, 0, 0, 1}");
        System.out.println("Gripe: " + resultado3[0]);
        System.out.println("Dengue: " + resultado3[1]);

        double[] entradaTeste4 = {0, 1, 1, 0};
        double[] resultado4 = mlp.prever(entradaTeste4);
        System.out.println();
        System.out.println("Entrada: {0, 1, 1, 0}");
        System.out.println("Gripe: " + resultado4[0]);
        System.out.println("Dengue: " + resultado4[1]);
    }
}