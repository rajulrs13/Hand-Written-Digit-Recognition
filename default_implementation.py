import dataset_loader
training_data, validation_data, test_data = dataset_loader.load_data_wrapper()
import neuralnet
x = int(input("Enter the no. of neurons in the hidden layer: "))
net = neuralnet.NeuralNet([784, x, 10])
e = int(input("Enter the no. of epochs: "))
s = int(input("Enter the size of each batch: "))
l = int(input("Enter the learning rate: "))
net.train(training_set = training_data, no_of_epochs = e, size_of_batch = s, learning_rate = l, test_set = test_data)
