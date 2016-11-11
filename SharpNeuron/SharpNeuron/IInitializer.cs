using SharpNeuron.Backpropagation;
using SharpNeuron.SOM;

namespace SharpNeuron
{
    /// <summary>
    /// Initializer interface. An initializer should define initialization methods for all concrete
    /// initializable layers and connectors.
    /// </summary>
    public interface IInitializer
    {
        /// <summary>
        /// Initializes bias values of activation neurons in an activation layer.
        /// </summary>
        /// <param name="activationLayer">
        /// The activation layer to initialize
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// If <c>activationLayer</c> is <c>null</c>
        /// </exception>
        void Initialize(ActivationLayer activationLayer);

        /// <summary>
        /// Initializes weights of all backpropagation synapses in a backpropagation connector.
        /// </summary>
        /// <param name="connector">
        /// The backpropagation connector to initialize.
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// If <c>connector</c> is <c>null</c>
        /// </exception>
        void Initialize(BackpropagationConnector connector);

        /// <summary>
        /// Initializes weights of all spatial synapses in a Kohonen connector.
        /// </summary>
        /// <param name="connector">
        /// The Kohonen connector to initialize.
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// If <c>connector</c> is <c>null</c>
        /// </exception>
        void Initialize(KohonenConnector connector);
    }
}