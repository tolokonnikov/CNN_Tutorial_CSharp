namespace CNN.Data
{
    public class DataReader
    {
        private readonly int rows = 28;
        private readonly int cols = 28;

        public List<Image> ReadData(string path)
        {
            var images = new List<Image>();

            try
            {
                using (var dataReader = new StreamReader(path))
                {
                    string line;

                    while ((line = dataReader.ReadLine()) != null)
                    {
                        var lineItems = line.Split(',');

                        double[,] data = new double[rows,cols];

                        int label = int.Parse(lineItems[0]);

                        int i = 1;
                        for (int row = 0; row < rows; row++)
                        {
                            for (int col = 0; col < cols; col++)
                            {
                                data[row,col] = int.Parse(lineItems[i]);
                                i++;
                            }
                        }

                        images.Add(new Image(data, label));
                    }
                }
            }
            catch (Exception e)
            {
                throw new ArgumentException($"File not found or invalid: {path}", e);
            }

            return images;
        }
    }
}
