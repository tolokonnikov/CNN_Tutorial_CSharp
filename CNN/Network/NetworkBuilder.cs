using CNN.Layers;

namespace CNN.Network
{

    public class NetworkBuilder
    {
        private NeuralNetwork net;
        private int _inputRows;
        private int _inputCols;
        private double _scaleFactor;
        private List<Layer> _layers;

        public NetworkBuilder(int inputRows, int inputCols, double scaleFactor)
        {
            _inputRows = inputRows;
            _inputCols = inputCols;
            _scaleFactor = scaleFactor;
            _layers = new List<Layer>();
        }

        public void AddConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED)
        {
            if (_layers.Count == 0)
            {
                _layers.Add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
            }
            else
            {
                Layer prev = _layers[_layers.Count - 1];
                _layers.Add(new ConvolutionLayer(filterSize, stepSize, prev.GetOutputLength(), prev.GetOutputRows(), prev.GetOutputCols(), SEED, numFilters, learningRate));
            }
        }

        public void AddMaxPoolLayer(int windowSize, int stepSize)
        {
            if (_layers.Count == 0)
            {
                _layers.Add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
            }
            else
            {
                Layer prev = _layers[_layers.Count - 1];
                _layers.Add(new MaxPoolLayer(stepSize, windowSize, prev.GetOutputLength(), prev.GetOutputRows(), prev.GetOutputCols()));
            }
        }

        public void AddFullyConnectedLayer(int outLength, double learningRate, long SEED)
        {
            if (_layers.Count == 0)
            {
                _layers.Add(new FullyConnectedLayer(_inputCols * _inputRows, outLength, SEED, learningRate));
            }
            else
            {
                Layer prev = _layers[_layers.Count - 1];
                _layers.Add(new FullyConnectedLayer(prev.GetOutputElements(), outLength, SEED, learningRate));
            }
        }

        public NeuralNetwork Build()
        {
            net = new NeuralNetwork(_layers, _scaleFactor);
            return net;
        }
    }

}
