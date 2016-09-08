using SharpNeuron.Backpropagation;
using System.Linq;

namespace SharpNeuron.Architects
{
    public class LSTM
    {
        public LSTM(params int[] layers)
        {
            var layerList = layers.ToList();
            var inputCount = layerList.First();
            var outputCount = layerList.Last();
            var hiddenLayerSizes = layerList.Skip(1).Take(layerList.Count - 2);
            foreach (var hiddenLayerNeuronCount in hiddenLayerSizes)
            {
                //generate memory blocks (memory cell and gates)
                var inputGate = new SigmoidLayer(hiddenLayerNeuronCount) { UseFixedBiasValues = true };
            }
        }
    }
}