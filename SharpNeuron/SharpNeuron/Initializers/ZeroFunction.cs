namespace SharpNeuron.Initializers
{
    /// <summary>
    /// An <see cref="IInitializer"/> using zero function
    /// </summary>
    public class ZeroFunction : ConstantFunction
    {
        /// <summary>
        /// Creates a new zero initializer function
        /// </summary>
        public ZeroFunction()
            : base(0d)
        {
        }
    }
}