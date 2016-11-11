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
            var hiddenLayerSizes = layerList.Skip(1).Take(layerList.Count - 2);
            var inputLayer = new LinearLayer(inputCount);
            var outputLayer = new SigmoidLayer(outputCount);
            var hiddenLayers = new List<SigmoidLayer>();
            ActivationLayer previousLayer = inputLayer;

            foreach (var hiddenLayerNeuronCount in hiddenLayerSizes)
            {
                var newHiddenLayer = new SigmoidLayer(hiddenLayerNeuronCount);
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