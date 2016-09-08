namespace SharpNeuron
{
    /// <summary>
    /// Mode of connection between layers.
    /// </summary>
    public enum ConnectionMode
    {
        /// <summary>
        /// A connection mode where all neurons of source layer are connected to all neurons of
        /// target layer
        /// </summary>
        AllToAll = 0,

        /// <summary>
        /// A connection mode where each neuron in source layer is connected to a single distinct
        /// neuron in the target layer. The source and target layers should have same number of
        /// neurons.
        /// </summary>
        OneToOne = 1,

        /// <summary>
        /// A connection mode where all neurons of source layer are connected to all neurons of
        /// the target layer, except that a neuron will not connect to itself.
        /// </summary>
        AllToElse = 2,
    }
}