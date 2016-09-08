using System;

namespace SharpNeuron.Backpropagation
{
    /// <summary>
    /// A connector represents a collection of synapses connecting two layers in a network.
    /// </summary>
    /// <typeparam name="TSourceLayer">Type of Source Layer</typeparam>
    /// <typeparam name="TTargetLayer">Type of Target Layer</typeparam>
    /// <typeparam name="TSynapse">Type of Synapse</typeparam>
    public class BackpropagationGate
        : Connector<ActivationLayer, ActivationLayer, BackpropagationSynapse>
    {
        public BackpropagationGate(ActivationLayer sourceLayer, ActivationLayer targetLayer)
            : this(sourceLayer, targetLayer, ConnectionMode.AllToAll)
        {
        }

        public BackpropagationGate(ActivationLayer sourceLayer, ActivationLayer targetLayer, ConnectionMode connectionMode)
            : base(sourceLayer, targetLayer, connectionMode)
        {
            ConstructSynapses();
        }

        private void ConstructSynapses()
        {
            throw new NotImplementedException();
        }

        public override void Initialize()
        {
            if (initializer != null)
            {
                initializer.Initialize(this);
            }
        }
    }
}