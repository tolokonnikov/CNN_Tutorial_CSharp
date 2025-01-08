namespace CNN.Layers
{
    public abstract class Layer
    {
        private Layer? _nextLayer;
        private Layer? _previousLayer;

        public Layer? NextLayer
        {
            get { return _nextLayer; }
            set { _nextLayer = value; }
        }

        public Layer? PreviousLayer
        {
            get { return _previousLayer; }
            set { _previousLayer = value; }
        }

        //public Layer get_nextLayer()
        //{
        //    return _nextLayer;
        //}

        //public void set_nextLayer(Layer _nextLayer)
        //{
        //    this._nextLayer = _nextLayer;
        //}

        //public Layer get_previousLayer()
        //{
        //    return _previousLayer;
        //}

        //public void set_previousLayer(Layer _previousLayer)
        //{
        //    this._previousLayer = _previousLayer;
        //}


        public abstract double[] GetOutput(List<double[,]> input);
        public abstract double[] GetOutput(double[] input);

        public abstract void BackPropagation(double[] dLdO);
        public abstract void BackPropagation(List<double[,]> dLdO);

        public abstract int GetOutputLength();
        public abstract int GetOutputRows();
        public abstract int GetOutputCols();
        public abstract int GetOutputElements();

        public double[] MatrixToVector(List<double[,]> input)
        {
            int totalSize = 0;
            foreach (var matrix in input)
            {
                totalSize += matrix.GetLength(0) * matrix.GetLength(1);
            }

            double[] output = new double[totalSize];
            int index = 0;

            foreach (var matrix in input)
            {
                int rows = matrix.GetLength(0);
                int cols = matrix.GetLength(1);

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        output[index++] = matrix[i, j];
                    }
                }
            }

            return output;
        }


        public List<double[,]> VectorToMatrix(double[] input, int length, int rows, int cols)
        {
            var output = new List<double[,]>();

            if (input.Length != length * rows * cols)
            {
                throw new ArgumentException("Input size does not match the specified dimensions.");
            }

            int index = 0;
            for (int l = 0; l < length; l++)
            {
                var matrix = new double[rows, cols];
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        matrix[i, j] = input[index++];
                    }
                }
                output.Add(matrix);
            }

            return output;
        }

    }

}
