using SharpNeuron.Backpropagation;
using SharpNeuron.SOM;

namespace SharpNeuron.Initializers
{
    /// <summary>
    /// An <see cref="IInitializer"/> using constant function
    /// </summary>
    public class ConstantFunction : IInitializer
    {
        private readonly double constant;

        /// <summary>
        /// Gets the initializer Constant
        /// </summary>
        /// <value>
        /// The constant with which each parameter (bias values and weights) is initialized
        /// </value>
        public double Constant
        {
            get { return constant; }
        }

        /// <summary>
        /// Creates a new constant function
        /// </summary>
        /// <param name="constant">
        /// The constant to use
        /// </param>
        public ConstantFunction(double constant)
        {
            this.constant = constant;
        }

        /// <summary>
        /// Initializes bias values of activation neurons in the activation layer.
        /// </summary>
        /// <param name="activationLayer">
        /// The activation layer to initialize
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If <c>activationLayer</c> is <c>null</c>
        /// </exception>
        public void Initialize(ActivationLayer activationLayer)
        {
            Helper.ValidateNotNull(activationLayer, "layer");
            foreach (ActivationNeuron neuron in activationLayer.Neurons)
            {
                neuron.Bias = constant;
            }
        }

        /// <summary>
        /// Initializes weights of all backpropagation synapses in the backpropagation connector.
        /// </summary>
        /// <param name="connector">
        /// The backpropagation connector to initialize.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If <c>connector</c> is <c>null</c>
        /// </exception>
        public void Initialize(BackpropagationConnector connector)
        {
            Helper.ValidateNotNull(connector, "connector");
            foreach (BackpropagationSynapse synapse in connector.Synapses)
            {
                synapse.Weight = constant;
            }
        }

        /// <summary>
        /// Initializes weights of all spatial synapses in a Kohonen connector.
        /// </summary>
        /// <param name="connector">
        /// The Kohonen connector to initialize.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If <c>connector</c> is <c>null</c>
        /// </exception>
        public void Initialize(KohonenConnector connector)
        {
            Helper.ValidateNotNull(connector, "connector");
            foreach (KohonenSynapse synapse in connector.Synapses)
            {
                synapse.Weight = constant;
            }
        }
    }
}