using SharpNeuron.Backpropagation;
using System.Collections.Generic;
using System.Linq;

namespace SharpNeuron.Architects
{
    public class Perceptron
    {
        public Perceptron(params int[] layers)
        {
            var layerList = layers.ToList();
            var inputCount = layerList.First();
            var outputCount = layerList.Last();
            var hiddenLayerCounts = layerList.Skip(1).Take(layerList.Count - 2);
            LinearLayer inputLayer = new LinearLayer(inputCount);
            SigmoidLayer outputLayer = new SigmoidLayer(outputCount);
            List<SigmoidLayer> hiddenLayers = new List<SigmoidLayer>();
            ActivationLayer previousLayer = inputLayer;
            foreach (var hiddenLayerCount in hiddenLayerCounts)
            {
                var hiddenLayer = new SigmoidLayer(hiddenLayerCount);
                hiddenLayers.Add(hiddenLayer);
                BackpropagationConnector.Connect(previousLayer, hiddenLayer);
                previousLayer = hiddenLayer;
            }
            BackpropagationConnector.Connect(previousLayer, outputLayer);
            var network = new BackpropagationNetwork(inputLayer, outputLayer);
            Network = network;
        }

        public Network Network { get; }
    }
}