using CNN.Data;
using CNN.Layers;

namespace CNN.Network
{
    public class NeuralNetwork
    {
        private List<Layer> _layers;
        private double scaleFactor;

        public NeuralNetwork(List<Layer> layers, double scaleFactor)
        {
            _layers = layers;
            this.scaleFactor = scaleFactor;
            LinkLayers();
        }

        private void LinkLayers()
        {
            if (_layers.Count <= 1)
            {
                return;
            }

            for (int i = 0; i < _layers.Count; i++)
            {
                if (i == 0)
                {
                    _layers[i].NextLayer = _layers[i + 1];
                }
                else if (i == _layers.Count - 1)
                {
                    _layers[i].PreviousLayer = _layers[i - 1];
                }
                else
                {
                    _layers[i].PreviousLayer = _layers[i - 1];
                    _layers[i].NextLayer = _layers[i + 1];
                }
            }
        }

        public double[] GetErrors(double[] networkOutput, int correctAnswer)
        {
            int numClasses = networkOutput.Length;

            double[] expected = new double[numClasses];
            expected[correctAnswer] = 1;

            return Add(networkOutput, Multiply(expected, -1));
        }

        private int GetMaxIndex(double[] inArray)
        {
            double max = 0;
            int index = 0;

            for (int i = 0; i < inArray.Length; i++)
            {
                if (inArray[i] >= max)
                {
                    max = inArray[i];
                    index = i;
                }
            }

            return index;
        }

        public int Guess(Image image)
        {
            List<double[,]> inList = new List<double[,]>
        {
            Multiply(image.Data, 1.0 / scaleFactor)
        };

            double[] outArray = _layers[0].GetOutput(inList);
            int guess = GetMaxIndex(outArray);

            return guess;
        }

        public float Test(List<Image> images)
        {
            int correct = 0;

            foreach (Image img in images)
            {
                int guess = Guess(img);

                if (guess == img.Label)
                {
                    correct++;
                }
            }

            return (float)correct / images.Count;
        }

        public void Train(List<Image> images)
        {
            foreach (Image img in images)
            {
                List<double[,]> inList = new List<double[,]>
            {
                Multiply(img.Data, 1.0 / scaleFactor)
            };

                double[] outArray = _layers[0].GetOutput(inList);
                double[] dldO = GetErrors(outArray, img.Label);

                _layers[_layers.Count - 1].BackPropagation(dldO);
            }
        }

        // Placeholder for the Add method
        private double[] Add(double[] a, double[] b)
        {
            // Implementation here (e.g., element-wise addition)
            int length = a.Length;
            double[] result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        // Placeholder for the Multiply method
        private double[,] Multiply(double[,] matrix, double scalar)
        {
            // Implementation for multiplying a matrix by a scalar
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            }

            return result;
        }

        // Placeholder for the Multiply method (array version)
        private double[] Multiply(double[] array, double scalar)
        {
            int length = array.Length;
            double[] result = new double[length];

            for (int i = 0; i < length; i++)
            {
                result[i] = array[i] * scalar;
            }

            return result;
        }
    }

}
