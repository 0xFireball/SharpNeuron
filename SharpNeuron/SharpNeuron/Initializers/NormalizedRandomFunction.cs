using SharpNeuron.Backpropagation;
using SharpNeuron.SOM;

namespace SharpNeuron.Initializers
{
    /// <summary>
    /// An <see cref="IInitializer"/> using Normalized Random function.
    /// </summary>
    public class NormalizedRandomFunction : IInitializer
    {
        /// <summary>
        /// Creates a new normalized random function
        /// </summary>
        public NormalizedRandomFunction()
        {
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
            Helper.ValidateNotNull(activationLayer, "activationLayer");

            int i = 0;
            var normalized = Helper.GetRandomVector(activationLayer.NeuronCount, 1d);
            foreach (ActivationNeuron neuron in activationLayer.Neurons)
            {
                neuron.Bias = normalized[i++];
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

            int i = 0;
            var normalized = Helper.GetRandomVector(connector.SynapseCount, 1d);
            foreach (BackpropagationSynapse synapse in connector.Synapses)
            {
                synapse.Weight = normalized[i++];
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

            int i = 0;
            var normalized = Helper.GetRandomVector(connector.SynapseCount, 1d);
            foreach (KohonenSynapse synapse in connector.Synapses)
            {
                synapse.Weight = normalized[i++];
            }
        }
    }
}