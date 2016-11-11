using SharpNeuron.Architects;
using System;

namespace SharpNeuron.Demo
{
    public class Program
    {
        public static void Main(string[] args)
        {
            //Run a XOR test
            Console.Write("Running XOR test...");

            var inputs = 2;
            var outputs = 1;

            var perceptron = new Perceptron(inputs, 3, outputs);
            var teacher = new PerceptronTeacher(perceptron);

            TrainingSet xorTrainingSet = new TrainingSet(2, 1);
            xorTrainingSet.Add(new TrainingSample(new double[2] { 0d, 0d }, new double[1] { 0d }));
            xorTrainingSet.Add(new TrainingSample(new double[2] { 0d, 1d }, new double[1] { 1d }));
            xorTrainingSet.Add(new TrainingSample(new double[2] { 1d, 0d }, new double[1] { 1d }));
            xorTrainingSet.Add(new TrainingSample(new double[2] { 1d, 1d }, new double[1] { 0d }));

            //Do 20K iterations unconditionalls
            teacher.Train(xorTrainingSet, new TeachingParameters { Iterations = 20000, ErrorThreshold = 0 });

            //Test network
            var outputSignal = perceptron.Run(new[] { 0d, 1d })[0];
            Console.WriteLine(outputSignal);
        }
    }
}