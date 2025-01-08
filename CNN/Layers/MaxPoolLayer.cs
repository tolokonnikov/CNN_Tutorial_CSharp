namespace CNN.Layers
{
    public class MaxPoolLayer : Layer
    {
        private int _stepSize;
        private int _windowSize;
        private int _inLength;
        private int _inRows;
        private int _inCols;

        private List<int[,]> _lastMaxRow;
        private List<int[,]> _lastMaxCol;

        public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols)
        {
            _stepSize = stepSize;
            _windowSize = windowSize;
            _inLength = inLength;
            _inRows = inRows;
            _inCols = inCols;
        }

        public List<double[,]> MaxPoolForwardPass(List<double[,]> input)
        {
            List<double[,]> output = new List<double[,]>();
            _lastMaxRow = new List<int[,]>();
            _lastMaxCol = new List<int[,]>();

            foreach (var item in input)
            {
                output.Add(Pool(item));
            }

            return output;
        }

        public double[,] Pool(double[,] input)
        {
            double[,] output = new double[GetOutputRows(), GetOutputCols()];
            int[,] maxRows = new int[GetOutputRows(), GetOutputCols()];
            int[,] maxCols = new int[GetOutputRows(), GetOutputCols()];

            for (int r = 0; r < GetOutputRows(); r += _stepSize)
            {
                for (int c = 0; c < GetOutputCols(); c += _stepSize)
                {
                    double max = 0.0;
                    maxRows[r, c] = -1;
                    maxCols[r, c] = -1;

                    for (int x = 0; x < _windowSize; x++)
                    {
                        for (int y = 0; y < _windowSize; y++)
                        {
                            if (max < input[r + x, c + y])
                            {
                                max = input[r + x, c + y];
                                maxRows[r, c] = r + x;
                                maxCols[r, c] = c + y;
                            }
                        }
                    }

                    output[r, c] = max;
                }
            }

            _lastMaxRow.Add(maxRows);
            _lastMaxCol.Add(maxCols);

            return output;
        }

        public override double[] GetOutput(List<double[,]> input)
        {
            List<double[,]> outputPool = MaxPoolForwardPass(input);
            return NextLayer.GetOutput(outputPool);
        }

        public override double[] GetOutput(double[] input)
        {
            List<double[,]> matrixList = VectorToMatrix(input, _inLength, _inRows, _inCols);
            return GetOutput(matrixList);
        }

        public override void BackPropagation(double[] dLdO)
        {
            List<double[,]> matrixList = VectorToMatrix(dLdO, GetOutputLength(), GetOutputRows(), GetOutputCols());
            BackPropagation(matrixList);
        }

        public override void BackPropagation(List<double[,]> dLdO)
        {
            List<double[,]> dXdL = new List<double[,]>();

            int l = 0;
            foreach (var array in dLdO)
            {
                double[,] error = new double[_inRows, _inCols];

                for (int r = 0; r < GetOutputRows(); r++)
                {
                    for (int c = 0; c < GetOutputCols(); c++)
                    {
                        int max_i = _lastMaxRow[l][r, c];
                        int max_j = _lastMaxCol[l][r, c];

                        if (max_i != -1)
                        {
                            error[max_i, max_j] += array[r, c];
                        }
                    }
                }

                dXdL.Add(error);
                l++;
            }

            if (PreviousLayer != null)
            {
                PreviousLayer.BackPropagation(dXdL);
            }
        }

        public override int GetOutputLength()
        {
            return _inLength;
        }

        public override int GetOutputRows()
        {
            return (_inRows - _windowSize) / _stepSize + 1;
        }

        public override int GetOutputCols()
        {
            return (_inCols - _windowSize) / _stepSize + 1;
        }

        public override int GetOutputElements()
        {
            return _inLength * GetOutputCols() * GetOutputRows();
        }
    }

}
