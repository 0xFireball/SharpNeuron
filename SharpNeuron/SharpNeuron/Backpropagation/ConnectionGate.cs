using System;
using System.Linq;

namespace SharpNeuron.Backpropagation
{
    public class ConnectionGate
    {
        public ConnectionGate(BackpropagationConnector connection, ConnectionGateType gateType)
        {
            switch (gateType)
            {
                case ConnectionGateType.Input:
                    if (connection.TargetLayer.NeuronCount != connection.SourceLayer.NeuronCount)
                    {
                        throw new InvalidOperationException("Source layer and target layer must be the same size in order to gate!");
                    }
                    for (var i = 0; i < connection.TargetLayer.NeuronCount; i++)
                    {
                        var neuron = connection.TargetLayer[i];
                        var gater = connection.SourceLayer[i];
                        for (var j = 0; i < neuron.SourceSynapses.Count; i++)
                        {
                            var gated = neuron.SourceSynapses[j];
                            if (connection.Synapses.Contains(gated))
                            {
                                //NOTE: Gate not yet finished!
                                gater.Gate(gated);
                            }
                        }
                    }
                    break;

                case ConnectionGateType.Output:
                    break;

                case ConnectionGateType.OneToOne:
                    break;

                default:
                    throw new NotImplementedException($"The gate type {gateType} is not implemented.");
            }
        }
    }
}