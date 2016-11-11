namespace SharpNeuron.Backpropagation
{
    /// <summary>
    /// A Backpropagation Connector is an <see cref="IConnector"/> which consists of a collection of
    /// backpropagation synapses connecting two activation layers.
    /// </summary>
    public class BackpropagationConnector
        : Connector<ActivationLayer, ActivationLayer, BackpropagationSynapse>
    {
        internal double momentum = 0.07d;

        /// <summary>
        /// Gets or sets the momentum (the tendency of synapses to retain their previous deltas)
        /// </summary>
        /// <value>
        /// The tendency of synapses to retain their previous weight change.
        /// </value>
        public double Momentum
        {
            get { return momentum; }
            set { momentum = value; }
        }

        /// <summary>
        /// Creates a new complete backpropagation connector between given layers.
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
        public BackpropagationConnector(ActivationLayer sourceLayer, ActivationLayer targetLayer)
            : this(sourceLayer, targetLayer, ConnectionMode.AllToAll)
        {
        }

        /// <summary>
        /// Creates a new Backpropagation connector between the given layers using the specified
        /// connection mode.
        /// </summary>
        /// <param name="sourceLayer">
        /// The source layer
        /// </param>
        /// <param name="targetLayer">
        /// The target layer
        /// </param>
        /// <param name="connectionMode">
        /// Connection mode to use
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If <c>sourceLayer</c> or <c>targetLayer</c> is <c>null</c>
        /// </exception>
        public BackpropagationConnector(ActivationLayer sourceLayer, ActivationLayer targetLayer, ConnectionMode connectionMode)
            : base(sourceLayer, targetLayer, connectionMode)
        {
            ConstructSynapses();
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

        /// <summary>
        /// Private helper to construct synapses between layers
        /// </summary>
        private void ConstructSynapses()
        {
            int i = 0;
            switch (connectionMode)
            {
                case ConnectionMode.AllToAll:
                    foreach (ActivationNeuron targetNeuron in targetLayer.Neurons)
                    {
                        foreach (ActivationNeuron sourceNeuron in sourceLayer.Neurons)
                        {
                            synapses[i++] = new BackpropagationSynapse(sourceNeuron, targetNeuron, this);
                        }
                    }

                    break;
                case ConnectionMode.OneToOne:
                    var sourceEnumerator = sourceLayer.Neurons.GetEnumerator();
                    var targetEnumerator = targetLayer.Neurons.GetEnumerator();
                    while (sourceEnumerator.MoveNext() && targetEnumerator.MoveNext())
                    {
                        synapses[i++] = new BackpropagationSynapse(
                            sourceEnumerator.Current, targetEnumerator.Current, this);
                    }

                    break;
                case ConnectionMode.AllToElse:
                    foreach (ActivationNeuron targetNeuron in targetLayer.Neurons)
                    {
                        foreach (ActivationNeuron sourceNeuron in sourceLayer.Neurons)
                        {
                            if (sourceNeuron == targetNeuron)
                            {
                                continue; //Skip connecting to self
                            }
                            synapses[i++] = new BackpropagationSynapse(sourceNeuron, targetNeuron, this);
                        }
                    }

                    break;
            }
        }

        public static BackpropagationConnector Connect(ActivationLayer sourceLayer, ActivationLayer targetLayer)
        {
            return new BackpropagationConnector(sourceLayer, targetLayer);
        }
    }
}