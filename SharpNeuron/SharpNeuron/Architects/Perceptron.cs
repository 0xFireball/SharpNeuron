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
                var newHiddenLayer = new SigmoidLayer(hiddenLayerCount);
                hiddenLayers.Add(newHiddenLayer);
                BackpropagationConnector.Connect(previousLayer, newHiddenLayer);
                previousLayer = newHiddenLayer;
            }
            BackpropagationConnector.Connect(previousLayer, outputLayer);
            var network = new BackpropagationNetwork(inputLayer, outputLayer);
            Network = network;
        }

        public double[] Run(params double[] inputData)
        {
            return Network.Run(inputData);
        }

        public BackpropagationNetwork Network { get; }
    }
}