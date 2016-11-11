using SharpNeuron.Initializers;

namespace SharpNeuron.SOM
{
    /// <summary>
    /// A Kohonen Connector is an <see cref="IConnector"/> consisting of a collection of Kohonen
    /// synapses connecting any layer to a Kohonen Layer.
    /// </summary>
    public class KohonenConnector : Connector<ILayer, KohonenLayer, KohonenSynapse>
    {
        /// <summary>
        /// Creates a new Kohonen connector between the given layers.
        /// </summary>
        /// <param name="sourceLayer">
        /// The source layer
        /// </param>
        /// <param name="targetLayer">
        /// The target layer
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If <c>sourceLayer</c> or <c>targetLayer</c> is <c>null</c>
        /// </exception>
        public KohonenConnector(ILayer sourceLayer, KohonenLayer targetLayer)
            : base(sourceLayer, targetLayer, ConnectionMode.AllToAll)
        {
            initializer = new RandomFunction();

            int i = 0;
            foreach (PositionNeuron targetNeuron in targetLayer.Neurons)
            {
                foreach (INeuron sourceNeuron in sourceLayer.Neurons)
                {
                    synapses[i++] = new KohonenSynapse(sourceNeuron, targetNeuron, this);
                }
            }
        }

        /// <summary>
        /// Initializes all synapses in the connector and makes them ready to undergo training
        /// freshly. (Adjusts the weights of synapses using the initializer)
        /// </summary>
        public override void Initialize()
        {
            if (initializer != null)
            {
                initializer.Initialize(this);
            }
        }
    }
}