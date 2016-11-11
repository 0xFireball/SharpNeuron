namespace SharpNeuron.LearningRateFunctions
{
    /// <summary>
    /// An abstract base class for a learning rate function.
    /// </summary>

    public abstract class AbstractFunction : ILearningRateFunction
    {
        /// <summary>
        /// Initial Learning Rate
        /// </summary>
        protected readonly double initialLearningRate;

        /// <summary>
        /// Final Learning Rate
        /// </summary>
        protected readonly double finalLearningRate;

        /// <summary>
        /// Gets the initial value of learning rate
        /// </summary>
        /// <value>
        /// Initial Learning Rate
        /// </value>
        public double InitialLearningRate
        {
            get { return initialLearningRate; }
        }

        /// <summary>
        /// Gets the final value of learning rate
        /// </summary>
        /// <value>
        /// Final Learning Rate
        /// </value>
        public double FinalLearningRate
        {
            get { return finalLearningRate; }
        }

        /// <summary>
        /// Constructs a new instance with the specified initial and final values of learning rate.
        /// </summary>
        /// <param name="initialLearningRate">
        /// Initial value learning rate
        /// </param>
        /// <param name="finalLearningRate">
        /// Final value learning rate
        /// </param>
        public AbstractFunction(double initialLearningRate, double finalLearningRate)
        {
            this.initialLearningRate = initialLearningRate;
            this.finalLearningRate = finalLearningRate;
        }

        /// <summary>
        /// Gets effective learning rate for current training iteration.
        /// </summary>
        /// <param name="currentIteration">
        /// Current training iteration
        /// </param>
        /// <param name="trainingEpochs">
        /// Total number of training epochs
        /// </param>
        /// <returns>
        /// The effective learning rate for current training iteration
        /// </returns>
        /// <exception cref="ArgumentException">
        /// If <c>trainingEpochs</c> is zero or negative
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// If <c>currentIteration</c> is negative or, if it is not less than <c>trainingEpochs</c>
        /// </exception>
        public abstract double GetLearningRate(int currentIteration, int trainingEpochs);
    }
}