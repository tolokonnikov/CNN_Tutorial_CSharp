namespace CNN.Data
{
    public class Image
    {
        private double[,] _data;
        private int _label;

        public double[,] Data
        {
            get { return _data; }
        }

        public int Label
        {
            get { return _label; }
        }

        public Image(double[,] data, int label)
        {
            _data = data;
            _label = label;
        }

        public override string ToString()
        {
            string s = _label + ", \n";

            for (int i = 0; i < _data.GetLength(0); i++)
            {
                for (int j = 0; j < _data.GetLength(1); j++)
                {
                    s += _data[i,j] + ", ";
                }
                s += "\n";
            }

            return s;
        }
    }
}
