namespace SharpNeuron.Architects
{
    public class PerceptronTeacher
    {
        public PerceptronTeacher(Perceptron studentNetwork)
        {
            Student = studentNetwork;
        }

        public void Train(TrainingSet data, TeachingParameters teachOptions)
        {
            //Set learning rate
            Student.Network.SetLearningRate(teachOptions.LearningRate);
            double error = 1;
            int teachingIterations;
            for (var i = 0; i < teachOptions.Iterations && error > teachOptions.ErrorThreshold; i++)
            {
                //Train on data set
                Student.Network.Learn(data, 1);
                error = Student.Network.MeanSquaredError;
                teachingIterations = i + 1; // i is zero based
            }
        }

        public Perceptron Student { get; }
    }
}