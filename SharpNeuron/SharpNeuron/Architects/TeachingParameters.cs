namespace SharpNeuron.Architects
{
    public class TeachingParameters
    {
        /// <summary>
        /// The maximum number of iterations to train the network for
        /// </summary>
        public int Iterations { get; set; } = 20000;

        /// <summary>
        /// The threshold of error at which to stop training the network. If the error of the network is below this threshold, teaching stops, even if not all of the iterations are finished. Set this to 0 to have the teacher always train for all the iterations
        /// </summary>
        public double ErrorThreshold { get; set; } = 0.005;

        /// <summary>
        /// The rate at which the network should learn
        /// </summary>
        public double LearningRate { get; set; } = 0.3;
    }
}