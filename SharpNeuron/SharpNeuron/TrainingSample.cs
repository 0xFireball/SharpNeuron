namespace SharpNeuron
{
    /// <summary>
    /// This class represents a training sample used to train a neural network
    /// </summary>
    public class TrainingSample
    {
        private readonly double[] inputVector;
        private readonly double[] outputVector;
        private readonly double[] normalizedInputVector;
        private readonly double[] normalizedOutputVector;
        private readonly int hashCode;

        /// <summary>
        /// Gets the value of input vector.
        /// </summary>
        /// <value>
        /// Input vector. It is never <c>null</c>.
        /// </value>
        public double[] InputVector
        {
            get { return inputVector; }
        }

        /// <summary>
        /// Gets the value of expected output vector
        /// </summary>
        /// <value>
        /// Output Vector. It is never <c>null</c>.
        /// </value>
        public double[] OutputVector
        {
            get { return outputVector; }
        }

        /// <summary>
        /// Gets the value of input vector in normalized form
        /// </summary>
        /// <value>
        /// Normalized Input Vector. It is never <c>null</c>.
        /// </value>
        public double[] NormalizedInputVector
        {
            get { return normalizedInputVector; }
        }

        /// <summary>
        /// Gets the value of output vector in normalized form.
        /// </summary>
        /// <value>
        /// Normalized Output Vector. It is never <c>null</c>.
        /// </value>
        public double[] NormalizedOutputVector
        {
            get { return normalizedOutputVector; }
        }

        /// <summary>
        /// Creates a new unsupervised training sample
        /// </summary>
        /// <param name="vector">
        /// The vector representing the unsupervised training sample
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// If vector is <c>null</c>
        /// </exception>
        public TrainingSample(double[] vector)
            : this(vector, new double[0])
        {
        }

        /// <summary>
        /// Creates a new training sample. The arguments are cloned into the training sample. So
        /// any modifications to the arguments will NOT be reflected in the training sample.
        /// </summary>
        /// <param name="inputVector">
        /// Input vector
        /// </param>
        /// <param name="outputVector">
        /// Expected output vector
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If any of the arguments is <c>null</c>
        /// </exception>
        public TrainingSample(double[] inputVector, double[] outputVector)
        {
            // Validate
            Helper.ValidateNotNull(inputVector, "inputVector");
            Helper.ValidateNotNull(outputVector, "outputVector");

            // Clone and initialize
            this.inputVector = (double[])inputVector.Clone();
            this.outputVector = (double[])outputVector.Clone();

            // Some neural networks require inputs in normalized form.
            // As an optimization measure, we normalize and store training samples
            normalizedInputVector = Helper.Normalize(inputVector);
            normalizedOutputVector = Helper.Normalize(outputVector);

            // Calculate the hash code
            hashCode = 0;
            for (int i = 0; i < inputVector.Length; i++)
            {
                hashCode ^= inputVector[i].GetHashCode();
            }
        }

        /// <summary>
        /// Determine whether the given object is equal to this instance
        /// </summary>
        /// <param name="obj">
        /// The object to compare with this instance
        /// </param>
        /// <returns>
        /// <c>true</c> if the given object is equal to this instance, <c>false</c> otherwise
        /// </returns>
        public override bool Equals(object obj)
        {
            if (obj is TrainingSample)
            {
                var sample = (TrainingSample)obj;
                int size;
                if ((size = sample.inputVector.Length) == inputVector.Length)
                {
                    for (int i = 0; i < size; i++)
                    {
                        //make sure they're unequal, use floating point comparison
                        if (System.Math.Abs(inputVector[i] - sample.inputVector[i]) > double.Epsilon)
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Serves as a hash function for a particular type
        /// </summary>
        /// <returns>
        /// The hash code for the current object
        /// </returns>
        public override int GetHashCode()
        {
            return hashCode;
        }
    }
}