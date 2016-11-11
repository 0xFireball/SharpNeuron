using SharpNeuron.Backpropagation;
using System.Collections.Generic;
using System.Linq;

namespace SharpNeuron.Architects
{
    public class LSTM
    {
        public LSTM(LSTMOptions options, params int[] layers)
        {
            var layerList = layers.ToList();
            var inputCount = layerList.First();
            var outputCount = layerList.Last();
            var hiddenLayerSizes = layerList.Skip(1).Take(layerList.Count - 2);

            var inputLayer = new LinearLayer(inputCount);
            var outputLayer = new SigmoidLayer(outputCount);
            var hiddenLayers = new List<SigmoidLayer>();
            ActivationLayer previousLayer = null;

            foreach (var hiddenLayerNeuronCount in hiddenLayerSizes)
            {
                //generate memory blocks (memory cell and gates)
                var inputGate = new SigmoidLayer(hiddenLayerNeuronCount) { UseFixedBiasValues = true };
                inputGate.SetBias(1);

                var forgetGate = new SigmoidLayer(hiddenLayerNeuronCount) { UseFixedBiasValues = true };
                forgetGate.SetBias(1);

                var memoryCell = new SigmoidLayer(hiddenLayerNeuronCount);

                var outputGate = new SigmoidLayer(hiddenLayerNeuronCount);
                outputGate.SetBias(1);

                hiddenLayers.Add(inputGate);
                hiddenLayers.Add(forgetGate);
                hiddenLayers.Add(memoryCell);
                hiddenLayers.Add(outputGate);

                //connections from input layer
                var input = BackpropagationConnector.Connect(inputLayer, memoryCell);
                BackpropagationConnector.Connect(inputLayer, inputGate);
                BackpropagationConnector.Connect(inputLayer, forgetGate);
                BackpropagationConnector.Connect(inputLayer, outputGate);

                //connections from previous memory block layer to this layer
                if (previousLayer != null)
                {
                    var cell = BackpropagationConnector.Connect(previousLayer, memoryCell);
                    BackpropagationConnector.Connect(previousLayer, inputGate);
                    BackpropagationConnector.Connect(previousLayer, forgetGate);
                    BackpropagationConnector.Connect(previousLayer, outputGate);
                }

                //connections from memory cell
                var output = BackpropagationConnector.Connect(memoryCell, outputLayer);

                //self-connection
                var self = BackpropagationConnector.Connect(memoryCell, memoryCell);

                //hidden to hidden recurrent connection
                if (options.HiddenToHidden)
                {
                    var cell = BackpropagationConnector.Connect(previousLayer, memoryCell);
                    BackpropagationConnector.Connect(previousLayer, inputGate);
                    BackpropagationConnector.Connect(previousLayer, forgetGate);
                    BackpropagationConnector.Connect(previousLayer, outputGate);
                }

                //out to hidden recurrent connection
                if (options.OutputToHidden)
                {
                    BackpropagationConnector.Connect(outputLayer, memoryCell);
                }

                //out to gates recurrent connection
                if (options.OutputToGates)
                {
                    BackpropagationConnector.Connect(outputLayer, inputGate);
                    BackpropagationConnector.Connect(outputLayer, outputGate);
                    BackpropagationConnector.Connect(outputLayer, forgetGate);
                }

                //peepholes
                //TODO: Make sure this is right
                if (options.Peepholes != null)
                {
                    BackpropagationConnector.Connect(inputGate, options.Peepholes);
                    BackpropagationConnector.Connect(forgetGate, options.Peepholes);
                    BackpropagationConnector.Connect(outputGate, options.Peepholes);
                }

                //gates
                //TODO: Implement gates!
                /*
                inputGate.gate(input, Layer.gateType.INPUT);
                forgetGate.gate(self, Layer.gateType.ONE_TO_ONE);
                outputGate.gate(output, Layer.gateType.OUTPUT);
                if (previousLayer != null)
                    inputGate.gate(cell, Layer.gateType.INPUT);
                */

                previousLayer = memoryCell;
            }
        }
    }
}