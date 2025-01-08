namespace CNN.Layers
{
    public class FullyConnectedLayer : Layer
    {
        private readonly double _leak = 0.01;

        private double[,] _weights;
        private int _inLength;
        private int _outLength;
        private double _learningRate;

        private double[] _lastZ;
        private double[] _lastX;
        private long _seed;

        public FullyConnectedLayer(int inLength, int outLength, long seed, double learningRate)
        {
            _inLength = inLength;
            _outLength = outLength;
            _seed = seed;
            _learningRate = learningRate;

            _weights = new double[_inLength, outLength];

            SetRandomWeights();
        }

        public double[] FullyConnectedForwardPass(double[] input)
        {
            _lastX = input;

            double[] z = new double[_outLength];
            double[] output = new double[_outLength];

            for (int i = 0; i < _inLength; i++)
            {
                for (int j = 0; j < _outLength; j++)
                {
                    z[j] += input[i] * _weights[i,j];
                }
            }

            _lastZ = z;

            for (int j = 0; j < _outLength; j++)
            {
                output[j] = ReLu(z[j]);
            }

            return output;
        }

        public override double[] GetOutput(List<double[,]> input)
        {
            double[] vector = MatrixToVector(input);
            return GetOutput(vector);
        }

        public override double[] GetOutput(double[] input)
        {
            double[] forwardPass = FullyConnectedForwardPass(input);

            if (NextLayer != null)
            {
                return NextLayer.GetOutput(forwardPass);
            }
            else
            {
                return forwardPass;
            }
        }

        public override void BackPropagation(double[] dLdO)
        {
            double[] dLdX = new double[_inLength];

            for (int k = 0; k < _inLength; k++)
            {
                double dLdXSum = 0;

                for (int j = 0; j < _outLength; j++)
                {
                    double dOdz = DerivativeReLu(_lastZ[j]);
                    double dzdw = _lastX[k];
                    double dzdx = _weights[k,j];

                    double dLdw = dLdO[j] * dOdz * dzdw;

                    _weights[k,j] -= dLdw * _learningRate;

                    dLdXSum += dLdO[j] * dOdz * dzdx;
                }

                dLdX[k] = dLdXSum;
            }

            if (PreviousLayer != null)
            {
                PreviousLayer.BackPropagation(dLdX);
            }
        }

        public override void BackPropagation(List<double[,]> dLdO)
        {
            double[] vector = MatrixToVector(dLdO);
            BackPropagation(vector);
        }

        public override int GetOutputLength() => 0;

        public override int GetOutputRows() => 0;

        public override int GetOutputCols() => 0;

        public override int GetOutputElements() => _outLength;

        private void SetRandomWeights()
        {
            var random = new Random((int)_seed);

            for (int i = 0; i < _inLength; i++)
            {
                for (int j = 0; j < _outLength; j++)
                {
                    _weights[i,j] = NextGaussian(random);
                }
            }
        }

        private double ReLu(double input) => input > 0 ? input : 0;

        private double DerivativeReLu(double input) => input > 0 ? 1 : _leak;

        private double NextGaussian(Random random)
        {
            // Box-Muller transform for Gaussian distribution
            double u1 = 1.0 - random.NextDouble(); // [0,1) -> (0,1]
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
