using System;
using System.Runtime.Serialization;

namespace SharpNeuron.Backpropagation
{
    /// <summary>
    /// Activation Layer is a layer of activation neurons.
    /// </summary>
    public abstract class ActivationLayer : Layer<ActivationNeuron>
    {
        protected bool useFixedBiasValues = false;

        /// <summary>
        /// Gets or sets a boolean representing whether to use fixed neuron bias values. If this is false, the bias value will change during learning.
        /// </summary>
        /// <value>
        /// A boolean indicating whether bias values of activation neurons learn while training.
        /// </value>
        public bool UseFixedBiasValues
        {
            get { return useFixedBiasValues; }
            set { useFixedBiasValues = value; }
        }

        /// <summary>
        /// Constructs an instance of activation Layer
        /// </summary>
        /// <param name="neuronCount">
        /// The number of neurons in the layer
        /// </param>
        /// <exception cref="ArgumentException">
        /// If <c>neuronCount</c> is zero or negative
        /// </exception>
        protected ActivationLayer(int neuronCount)
            : base(neuronCount)
        {
            for (int i = 0; i < neuronCount; i++)
            {
                neurons[i] = new ActivationNeuron(this);
            }
        }

        /// <summary>
        /// Initializes all neurons and makes them ready to undergo training freshly.
        /// </summary>
        public override void Initialize()
        {
            if (initializer != null)
            {
                initializer.Initialize(this);
            }
        }

        /// <summary>
        /// Sets neuron errors as the difference between actual and expected outputs
        /// </summary>
        /// <param name="expectedOutput">
        /// Expected output vector
        /// </param>
        /// <returns>
        /// Mean squared error
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// If <c>expectedOutput</c> is <c>null</c>
        /// </exception>
        /// <exception cref="ArgumentException">
        /// If length of <c>expectedOutput</c> is different from the number of neurons
        /// </exception>
        public double SetErrors(double[] expectedOutput)
        {
            // Validate
            Helper.ValidateNotNull(expectedOutput, "expectedOutput");
            if (expectedOutput.Length != neurons.Length)
            {
                throw new ArgumentException("Length of ouput array should be same as neuron count", "expectedOutput");
            }

            // Set errors, evaluate mean squared error
            double meanSquaredError = 0d;
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].Error = expectedOutput[i] - neurons[i].Output;
                meanSquaredError += neurons[i].Error * neurons[i].Error;
            }
            return meanSquaredError;
        }

        /// <summary>
        /// Evaluate errors at all neurons in the layer
        /// </summary>
        public void EvaluateErrors()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].EvaluateError();
            }
        }

        /// <summary>
        /// Activation function used by all neurons in this layer
        /// </summary>
        /// <param name="input">
        /// Current input to the neuron
        /// </param>
        /// <param name="previousOutput">
        /// The previous output at the neuron
        /// </param>
        /// <returns>
        /// The activated value
        /// </returns>
        public abstract double Activate(double input, double previousOutput);

        /// <summary>
        /// Derivative function used by all neurons in this layer
        /// </summary>
        /// <param name="input">
        /// Current input to the neuron
        /// </param>
        /// <param name="output">
        /// Current output (activated) at the neuron
        /// </param>
        /// <returns>
        /// The result of derivative of activation function
        /// </returns>
        public abstract double Derivative(double input, double output);
    }
}