using System;

namespace SharpNeuron.SOM.NeighborhoodFunctions
{
    /// <summary>
    /// Mexican Hat Neighborhood Function is the normalized second derivative of a Gaussian function.
    /// It is a continuous function with neighborhood value decreasing from unity at the winner to
    /// a negative value at a certain point (forming an inhibitory influence) and then gradually
    /// increasing to zero.
    /// </summary>
    public sealed class MexicanHatFunction : INeighborhoodFunction
    {
        /*
         *  Mexican Hat Function = a * (1 - ((x-b)/c)square) * Exp( - 1/2 * ((x-b)/c)square)
         *
         *  The parameter 'a' is the height of the curve's peak, 'b' is the position of the center of
         *  the peak, and 'c' controls the width of the bell shape.
         *
         *  For a Mexican Hat Neighborhood function,
         *  a = unity (the neighborhood at the winner)
         *  b = winner position
         *  c = depends on training progress.
         *
         *  Initial value of c is obtained from the user (as learning radius)
         *  Note that, (x-b)square denotes the euclidean distance between winner neuron 'b' and neuron 'x'
         *
         *  (Mexican hat function) vs (Hamming distance)
         *                         _
         *                        / \
         *              _____    |   |    _____
         *                   \__/     \__/
         *                         .
         *                       Winner
         */

        private readonly double sigma = 0d;

        /// <summary>
        /// Gets the value of sigma
        /// </summary>
        /// <value>
        /// Initial value of sigma
        /// </value>
        public double Sigma
        {
            get { return sigma; }
        }

        /// <summary>
        /// Creates a new Mexican Hat Neighborhood Function
        /// </summary>
        /// <param name="learningRadius">
        /// Initial Learning Radius
        /// </param>
        public MexicanHatFunction(int learningRadius)
        {
            // Full Width at Half Maximum for a Mexican Hat curve
            //        = 1.2518753 * sigma
            // Full Width at Half Maximum (FWHM) is nothing but learning diameter
            // so, learning radius = 0.62593765 * sigma

            sigma = learningRadius / 0.6259d;
        }

        /// <summary>
        /// Determines the neighborhood of every neuron in the given Kohonen layer with respect to
        /// winner neuron using Mexican Hat function
        /// </summary>
        /// <param name="layer">
        /// The Kohonen Layer containing neurons
        /// </param>
        /// <param name="currentIteration">
        /// Current training iteration
        /// </param>
        /// <param name="trainingEpochs">
        /// Total number of training epochs
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// If <c>layer</c> is <c>null</c>
        /// </exception>
        /// <exception cref="ArgumentException">
        /// If <c>trainingEpochs</c> is zero or negative
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// If <c>currentIteration</c> is negative or, if it is not less than <c>trainingEpochs</c>
        /// </exception>
        public void EvaluateNeighborhood(KohonenLayer layer, int currentIteration, int trainingEpochs)
        {
            Helper.ValidateNotNull(layer, "layer");
            Helper.ValidatePositive(trainingEpochs, "trainingEpochs");
            Helper.ValidateWithinRange(currentIteration, 0, trainingEpochs - 1, "currentIteration");

            // Winner co-ordinates
            int winnerX = layer.Winner.Coordinate.X;
            int winnerY = layer.Winner.Coordinate.Y;

            // Layer width and height
            int layerWidth = layer.Size.Width;
            int layerHeight = layer.Size.Height;

            // Optimization: Pre-calculated 2-Sigma-Square (1e-9 to make sure it is non-zero)
            double sigmaSquare = sigma * sigma + 1e-9;

            // Evaluate and update neighborhood value of each neuron
            foreach (PositionNeuron neuron in layer.Neurons)
            {
                var dx = Math.Abs(winnerX - neuron.Coordinate.X);
                var dy = Math.Abs(winnerY - neuron.Coordinate.Y);

                if (layer.IsRowCircular)
                {
                    dx = Math.Min(dx, layerWidth - dx);
                }
                if (layer.IsColumnCircular)
                {
                    dy = Math.Min(dy, layerHeight - dy);
                }

                double dxSquare = dx * dx;
                double dySquare = dy * dy;
                if (layer.Topology == LatticeTopology.Hexagonal)
                {
                    if (dy % 2 == 1)
                    {
                        dxSquare += 0.25 + (((neuron.Coordinate.X > winnerX) == (winnerY % 2 == 0)) ? dx : -dx);
                    }
                    dySquare *= 0.75;
                }
                double distanceBySigmaSquare = (dxSquare + dySquare) / sigmaSquare;
                neuron.neighborhoodValue = (1 - distanceBySigmaSquare) * Math.Exp(-distanceBySigmaSquare / 2);
            }
        }
    }
}