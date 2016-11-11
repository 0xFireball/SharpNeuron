using System.Collections.Generic;

namespace SharpNeuron.Backpropagation
{
    /// <summary>
    /// Activation Neuron is a buiding block of a back-propagation neural network.
    /// </summary>
    public class ActivationNeuron : INeuron
    {
        protected double input;
        protected double output;
        protected double error;
        protected double bias;

        private readonly IList<ISynapse> sourceSynapses = new List<ISynapse>();
        private readonly IList<ISynapse> targetSynapses = new List<ISynapse>();
        private readonly IList<ISynapse> gateSynapses = new List<ISynapse>();

        private ActivationLayer parent;

        /// <summary>
        /// Gets or sets the neuron input.
        /// </summary>
        /// <value>
        /// Input to the neuron. For input neurons, this value is specified by user, whereas other
        /// neurons will have their inputs updated when the source synapses propagate
        /// </value>
        public double Input
        {
            get { return input; }
            set { input = value; }
        }

        /// <summary>
        /// Gets the output of the neuron.
        /// </summary>
        /// <value>
        /// Neuron Output
        /// </value>
        public double Output
        {
            get { return output; }
            set { output = value; }
        }

        /// <summary>
        /// Gets the neuron error
        /// </summary>
        /// <value>
        /// Neuron Error
        /// </value>
        public double Error
        {
            get { return error; }
            set { error = value; }
        }

        /// <summary>
        /// Gets or sets the neuron bias
        /// </summary>
        /// <value>
        /// The current value of neuron bias
        /// </value>
        public double Bias
        {
            get { return bias; }
            set { bias = value; }
        }

        /// <summary>
        /// Gets the list of source synapses associated with this neuron
        /// </summary>
        /// <value>
        /// A list of source synapses. It can neither be <c>null</c>, nor contain <c>null</c> elements.
        /// </value>
        public IList<ISynapse> SourceSynapses
        {
            get { return sourceSynapses; }
        }

        /// <summary>
        /// Gets the list of target synapses associated with this neuron
        /// </summary>
        /// <value>
        /// A list of target synapses. It can neither be <c>null</c>, nor contains <c>null</c> elements.
        /// </value>
        public IList<ISynapse> TargetSynapses
        {
            get { return targetSynapses; }
        }

        /// <summary>
        /// Gets the list of gate synapses associated with this neuron
        /// </summary>
        /// <value>
        /// A list of gate synapses. It can neither be <c>null</c>, nor contains <c>null</c> elements.
        /// </value>
        public IList<ISynapse> GateSynapses
        {
            get { return gateSynapses; }
        }

        /// <summary>
        /// Gets the parent layer containing this neuron
        /// </summary>
        /// <value>
        /// The parent layer containing this neuron. It is never <c>null</c>
        /// </value>
        public ActivationLayer Parent
        {
            get { return parent; }
        }

        ILayer INeuron.Parent
        {
            get { return parent; }
        }

        /// <summary>
        /// Create a new activation neuron
        /// </summary>
        /// <param name="parent">
        /// The parent layer containing this neuron
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// If <c>parent</c> is <c>null</c>
        /// </exception>
        public ActivationNeuron(ActivationLayer parent, bool useRandomBias = false)
        {
            Helper.ValidateNotNull(parent, "parent");

            input = 0d;
            output = 0d;
            error = 0d;
            bias = 0d;
            if (useRandomBias)
            {
                bias = RandomProvider.RNG.NextDouble() * 0.2 - 0.1;
            }
            this.parent = parent;
        }

        /// <summary>
        /// Obtains input from source synapses and activates to update the output
        /// </summary>
        public void Run()
        {
            if (sourceSynapses.Count > 0)
            {
                input = 0d;
                for (int i = 0; i < sourceSynapses.Count; i++)
                {
                    sourceSynapses[i].Propagate();
                }
            }
            output = parent.Activate(bias + input, output);
        }

        /// <summary>
        /// Backpropagates the target synapses and evaluates the error
        /// </summary>
        public void EvaluateError()
        {
            if (targetSynapses.Count > 0)
            {
                error = 0d;
                foreach (BackpropagationSynapse synapse in targetSynapses)
                {
                    synapse.Backpropagate();
                }
            }
            error *= parent.Derivative(input, output);
        }

        /// <summary>
        /// Optimizes the bias value (if not <c>UseFixedBiasValues</c>) and the weights of all the
        /// source synapses using back propagation algorithm
        /// </summary>
        /// <param name="learningRate">
        /// The current learning rate (this depends on training progress as well)
        /// </param>
        public void Learn(double learningRate)
        {
            if (!parent.UseFixedBiasValues)
            {
                bias += learningRate * error;
            }
            for (int i = 0; i < sourceSynapses.Count; i++)
            {
                sourceSynapses[i].OptimizeWeight(learningRate);
            }
        }

        public void Gate(ISynapse connection)
        {
            //Add connection to gated list
            gateSynapses.Add(connection);

            var neuron = connection.TargetNeuron;

            //keep track
            //TODO: implement trace
        }
    }
}