using CNN.Data;
using CNN.Network;

namespace CNN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, CNN World!");

            long SEED = 123;

            Console.WriteLine("Starting data loading...");

            List<Image> imagesTest = new DataReader().ReadData("Data/mnist_test.csv");
            List<Image> imagesTrain = new DataReader().ReadData("Data/mnist_train.csv");

            Console.WriteLine($"Images Train size: {imagesTrain.Count}");
            Console.WriteLine($"Images Test size: {imagesTest.Count}");

            NetworkBuilder builder = new NetworkBuilder(28, 28, 256 * 100);
            builder.AddConvolutionLayer(8, 5, 1, 0.1, SEED);
            builder.AddMaxPoolLayer(3, 2);
            builder.AddFullyConnectedLayer(10, 0.1, SEED);

            NeuralNetwork net = builder.Build();

            float rate = net.Test(imagesTest);
            Console.WriteLine($"Pre training success rate: {rate}");

            int epochs = 3;

            for (int i = 0; i < epochs; i++)
            {
                imagesTrain.Shuffle();
                net.Train(imagesTrain);
                rate = net.Test(imagesTest);
                Console.WriteLine($"Success rate after round {i}: {rate}");
            }
        }
    }
}
