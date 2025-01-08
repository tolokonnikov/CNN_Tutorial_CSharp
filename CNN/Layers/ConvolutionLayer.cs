using CNN.Data;

namespace CNN.Layers
{
    public class ConvolutionLayer : Layer
    {
        private long SEED;

        private List<double[,]> _filters;
        private int _filterSize;
        private int _stepSize;

        private int _inLength;
        private int _inRows;
        private int _inCols;
        private double _learningRate;

        private List<double[,]> _lastInput;

        public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows, int inCols, long seed, int numFilters, double learningRate)
        {
            _filterSize = filterSize;
            _stepSize = stepSize;
            _inLength = inLength;
            _inRows = inRows;
            _inCols = inCols;
            SEED = seed;
            _learningRate = learningRate;

            GenerateRandomFilters(numFilters);
        }

        private void GenerateRandomFilters(int numFilters)
        {
            _filters = new List<double[,]>();
            Random random = new Random((int)SEED);

            for (int n = 0; n < numFilters; n++)
            {
                double[,] filter = new double[_filterSize, _filterSize];

                for (int i = 0; i < _filterSize; i++)
                {
                    for (int j = 0; j < _filterSize; j++)
                    {
                        filter[i, j] = random.NextDouble() * 2 - 1; // Random values between -1 and 1
                    }
                }

                _filters.Add(filter);
            }
        }

        public List<double[,]> ConvolutionForwardPass(List<double[,]> input)
        {
            _lastInput = input;
            var output = new List<double[,]>();

            foreach (var inputMatrix in input)
            {
                foreach (var filter in _filters)
                {
                    output.Add(Convolve(inputMatrix, filter, _stepSize));
                }
            }

            return output;
        }

        private double[,] Convolve(double[,] input, double[,] filter, int stepSize)
        {
            int outRows = (input.GetLength(0) - filter.GetLength(0)) / stepSize + 1;
            int outCols = (input.GetLength(1) - filter.GetLength(1)) / stepSize + 1;

            double[,] output = new double[outRows, outCols];

            for (int i = 0, outRow = 0; i <= input.GetLength(0) - filter.GetLength(0); i += stepSize, outRow++)
            {
                for (int j = 0, outCol = 0; j <= input.GetLength(1) - filter.GetLength(1); j += stepSize, outCol++)
                {
                    double sum = 0.0;

                    for (int x = 0; x < filter.GetLength(0); x++)
                    {
                        for (int y = 0; y < filter.GetLength(1); y++)
                        {
                            sum += filter[x, y] * input[i + x, j + y];
                        }
                    }

                    output[outRow, outCol] = sum;
                }
            }

            return output;
        }

        public double[,] SpaceArray(double[,] input)
        {
            if (_stepSize == 1)
                return input;

            int outRows = (input.GetLength(0) - 1) * _stepSize + 1;
            int outCols = (input.GetLength(1) - 1) * _stepSize + 1;

            double[,] output = new double[outRows, outCols];

            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    output[i * _stepSize, j * _stepSize] = input[i, j];
                }
            }

            return output;
        }

        public override double[] GetOutput(List<double[,]> input)
        {
            var output = ConvolutionForwardPass(input);
            return NextLayer?.GetOutput(output) ?? MatrixToVector(output);
        }

        public override double[] GetOutput(double[] input)
        {
            var matrixInput = VectorToMatrix(input, _inLength, _inRows, _inCols);
            return GetOutput(matrixInput);
        }

        public override void BackPropagation(double[] dLdO)
        {
            var matrixInput = VectorToMatrix(dLdO, _inLength, _inRows, _inCols);
            BackPropagation(matrixInput);
        }

        public override void BackPropagation(List<double[,]> dLdO)
        {
            var filtersDelta = new List<double[,]>();
            var dLdOPreviousLayer = new List<double[,]>();

            for (int f = 0; f < _filters.Count; f++)
            {
                filtersDelta.Add(new double[_filterSize, _filterSize]);
            }

            for (int i = 0; i < _lastInput.Count; i++)
            {
                double[,] errorForInput = new double[_inRows, _inCols];

                for (int f = 0; f < _filters.Count; f++)
                {
                    var currFilter = _filters[f];
                    var error = dLdO[i * _filters.Count + f];

                    var spacedError = SpaceArray(error);
                    var dLdF = Convolve(_lastInput[i], spacedError, 1);

                    var delta = MatrixUtility.Multiply(dLdF, _learningRate * -1);
                    var newTotalDelta = MatrixUtility.Add(filtersDelta[f], delta);
                    filtersDelta[f] = newTotalDelta;

                    var flippedError = FlipArrayVertical(FlipArrayHorizontal(spacedError));
                    errorForInput = MatrixUtility.Add(errorForInput, FullConvolve(currFilter, flippedError));
                }

                dLdOPreviousLayer.Add(errorForInput);
            }

            for (int f = 0; f < _filters.Count; f++)
            {
                _filters[f] = MatrixUtility.Add(filtersDelta[f], _filters[f]);
            }

            PreviousLayer?.BackPropagation(dLdOPreviousLayer);
        }

        public double[,] FlipArrayHorizontal(double[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            double[,] output = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[rows - i - 1, j] = array[i, j];
                }
            }

            return output;
        }

        public double[,] FlipArrayVertical(double[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            double[,] output = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, cols - j - 1] = array[i, j];
                }
            }

            return output;
        }

        private double[,] FullConvolve(double[,] input, double[,] filter)
        {
            int outRows = input.GetLength(0) + filter.GetLength(0) - 1;
            int outCols = input.GetLength(1) + filter.GetLength(1) - 1;

            double[,] output = new double[outRows, outCols];

            for (int i = 0; i < outRows; i++)
            {
                for (int j = 0; j < outCols; j++)
                {
                    double sum = 0.0;

                    for (int x = 0; x < filter.GetLength(0); x++)
                    {
                        for (int y = 0; y < filter.GetLength(1); y++)
                        {
                            int inputRowIndex = i - x;
                            int inputColIndex = j - y;

                            if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < input.GetLength(0) && inputColIndex < input.GetLength(1))
                            {
                                sum += filter[x, y] * input[inputRowIndex, inputColIndex];
                            }
                        }
                    }

                    output[i, j] = sum;
                }
            }

            return output;
        }

        public override int GetOutputLength()
        {
            return _filters.Count * _inLength;
        }

        public override int GetOutputRows()
        {
            return (_inRows - _filterSize) / _stepSize + 1;
        }

        public override int GetOutputCols()
        {
            return (_inCols - _filterSize) / _stepSize + 1;
        }

        public override int GetOutputElements()
        {
            return GetOutputCols() * GetOutputRows() * GetOutputLength();
        }
    }

}
