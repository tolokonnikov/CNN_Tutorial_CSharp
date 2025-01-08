namespace CNN.Data
{
    public static class MatrixUtility
    {
        public static double[,] Add(double[,] a, double[,] b)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            {
                throw new ArgumentException("Масиви повинні мати однакові розміри.");
            }

            int rows = a.GetLength(0);
            int cols = a.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }

            return result;
        }

        public static double[] Add(double[] a, double[] b)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        public static double[,] Multiply(double[,] a, double scalar)
        {
            int rows = a.GetLength(0);
            int cols = a.GetLength(1);

            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] * scalar;
                }
            }

            return result;
        }


        public static double[] Multiply(double[] a, double scalar)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * scalar;
            }
            return result;
        }
    }
}
