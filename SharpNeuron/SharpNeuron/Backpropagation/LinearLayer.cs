namespace SharpNeuron.Backpropagation
{
    /// <summary>
    /// An <see cref="ActivationLayer"/> using linear activation function
    /// </summary>
    public class LinearLayer : ActivationLayer
    {
        /// <summary>
        /// Constructs a new LinearLayer containing specified number of neurons
        /// </summary>
        /// <param name="neuronCount">
        /// The number of neurons
        /// </param>
        /// <exception cref="ArgumentException">
        /// If <c>neuronCount</c> is zero or negative
        /// </exception>
        public LinearLayer(int neuronCount)
            : base(neuronCount)
        {
            useFixedBiasValues = true;
        }

        /// <summary>
        /// Linear activation function
        /// </summary>
        /// <param name="input">
        /// Current input to the neuron
        /// </param>
        /// <param name="previousOutput">
        /// The previous output at the neuron
        /// </param>
        /// <returns>
        /// The activated value
        /// </returns>
        public override double Activate(double input, double previousOutput)
        {
            return input;
        }

        /// <summary>
        /// Derivative of linear function
        /// </summary>
        /// <param name="input">
        /// Current input to the neuron
        /// </param>
        /// <param name="output">
        /// Current output (activated) at the neuron
        /// </param>
        /// <returns>
        /// The result of derivative of activation function
        /// </returns>
        public override double Derivative(double input, double output)
        {
            return 1d;
        }
    }
}