using SharpNeuron.Backpropagation;
using SharpNeuron.SOM;

namespace SharpNeuron.Initializers
{
    /// <summary>
    /// An <see cref="IInitializer"/> using random function
    /// </summary>

    public class RandomFunction : IInitializer
    {
        private readonly double minLimit;
        private readonly double maxLimit;

        /// <summary>
        /// Gets the minimum random limit
        /// </summary>
        /// <value>
        /// Minimum limit to the random initial values
        /// </value>
        public double MinLimit
        {
            get { return minLimit; }
        }

        /// <summary>
        /// Gets the maximum random limit
        /// </summary>
        /// <value>
        /// Maximum limit to the random initial values
        /// </value>
        public double MaxLimit
        {
            get { return maxLimit; }
        }

        /// <summary>
        /// Creates a new random initialization function which uses random values from 0 to 1
        /// </summary>
        public RandomFunction()
            : this(0, 1)
        {
        }

        /// <summary>
        /// Creates a new random initialization function using random values between the specified
        /// limits.
        /// </summary>
        /// <param name="minLimit">
        /// The minimum limit
        /// </param>
        /// <param name="maxLimit">
        /// The maximum limit
        /// </param>
        public RandomFunction(double minLimit, double maxLimit)
        {
            this.minLimit = minLimit;
            this.maxLimit = maxLimit;
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
                neuron.Bias = Helper.GetRandom(minLimit, maxLimit);
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
                synapse.Weight = Helper.GetRandom(minLimit, maxLimit);
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
                synapse.Weight = Helper.GetRandom(minLimit, maxLimit);
            }
        }
    }
}