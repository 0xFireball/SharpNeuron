using System;

namespace SharpNeuron
{
    internal static class RandomProvider
    {
        public static Random RNG { get; set; }

        static RandomProvider()
        {
            RNG = new Random();
        }
    }
}