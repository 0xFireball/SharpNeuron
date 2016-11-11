using SharpNeuron.Backpropagation;
using SharpNeuron.SOM;
using System;

namespace SharpNeuron.Initializers
{
    /// <summary>
    /// An <see cref="IInitializer"/> using Nguyen Widrow function.
    /// </summary>
    public class NguyenWidrowFunction : IInitializer
    {
        private readonly double outputRange;

        /// <summary>
        /// Gets the output range
        /// </summary>
        /// <value>
        /// The range of values, that network output takes
        /// </value>
        public double OutputRange
        {
            get { return outputRange; }
        }

        /// <summary>
        /// Creates a new NGuyen Widrow Initialization function
        /// </summary>
        public NguyenWidrowFunction()
            : this(1d)
        {
        }

        /// <summary>
        /// Creates a new NGuyen Widrow function using the given output range
        /// </summary>
        /// <param name="outputRange">
        /// the range of values, that output of a neuron can take (i.e. maximum minus minimum)
        /// </param>
        public NguyenWidrowFunction(double outputRange)
        {
            this.outputRange = outputRange;
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

            int hiddenNeuronCount = 0;
            foreach (IConnector targetConnector in activationLayer.TargetConnectors)
            {
                hiddenNeuronCount += targetConnector.TargetLayer.NeuronCount;
            }

            var nGuyenWidrowFactor = NGuyenWidrowFactor(activationLayer.NeuronCount, hiddenNeuronCount);

            foreach (ActivationNeuron neuron in activationLayer.Neurons)
            {
                neuron.Bias = Helper.GetRandom(-nGuyenWidrowFactor, nGuyenWidrowFactor);
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

            var nGuyenWidrowFactor = NGuyenWidrowFactor(
                connector.SourceLayer.NeuronCount, connector.TargetLayer.NeuronCount);

            int synapsesPerNeuron = connector.SynapseCount / connector.TargetLayer.NeuronCount;

            foreach (INeuron neuron in connector.TargetLayer.Neurons)
            {
                int i = 0;
                var normalizedVector = Helper.GetRandomVector(synapsesPerNeuron, nGuyenWidrowFactor);
                foreach (BackpropagationSynapse synapse in connector.GetSourceSynapses(neuron))
                {
                    synapse.Weight = normalizedVector[i++];
                }
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
            var nGuyenWidrowFactor = NGuyenWidrowFactor(
                connector.SourceLayer.NeuronCount, connector.TargetLayer.NeuronCount);

            int synapsesPerNeuron = connector.SynapseCount / connector.TargetLayer.NeuronCount;

            foreach (INeuron neuron in connector.TargetLayer.Neurons)
            {
                int i = 0;
                var normalizedVector = Helper.GetRandomVector(synapsesPerNeuron, nGuyenWidrowFactor);
                foreach (KohonenSynapse synapse in connector.GetSourceSynapses(neuron))
                {
                    synapse.Weight = normalizedVector[i++];
                }
            }
        }

        /// <summary>
        /// Private helper method to calculate Nguyen-Widrow factor
        /// </summary>
        /// <param name="inputNeuronCount">
        /// Number of input neurons
        /// </param>
        /// <param name="hiddenNeuronCount">
        /// Number of hidden neurons
        /// </param>
        /// <returns>
        /// The Nguyen-Widrow factor
        /// </returns>
        private double NGuyenWidrowFactor(int inputNeuronCount, int hiddenNeuronCount)
        {
            return 0.7d * Math.Pow(hiddenNeuronCount, (1d / inputNeuronCount)) / outputRange;
        }
    }
}