using SharpNeuron.Backpropagation;

namespace SharpNeuron.Architects
{
    public class LSTMOptions
    {
        //TODO: Peepholes?
        public ActivationLayer Peepholes { get; set; }

        public bool HiddenToHidden { get; set; }

        public bool OutputToHidden { get; set; }

        public bool OutputToGates { get; set; }

        public bool InputToOutput { get; set; }

        public LSTMOptions CreateDefault()
        {
            return new LSTMOptions();
        }
    }
}