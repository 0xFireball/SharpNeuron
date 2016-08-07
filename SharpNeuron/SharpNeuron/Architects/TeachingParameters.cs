namespace SharpNeuron.Architects
{
    public class TeachingParameters
    {
        public int Iterations { get; set; } = 20000;
        public double ErrorThreshold { get; set; } = 0.005;
        public double LearningRate { get; set; } = 0.3;
    }
}