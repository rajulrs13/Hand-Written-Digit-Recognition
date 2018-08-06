# Steps to Follow

1. Clone this repository or download all files.
2. Open a terminal or command prompt.
3. Navigate to the directory where you downloaded the files.
4. Open python terminal/ console by typing 
	>> python
5. Check for the version of python. As this is a python 3 code, it won't work on earlier versions (python 2.x.x).
6. Now type these commands one by one in the terminal.
	```
	>> import data_set_loader
	```

	"Load and Split the data into training and test sets"
	```
	>> training_data, validation_data, test_data = data_set_loader.load_data_wrapper()
	```
	
	"Import the main file containing NeuralNet class"
	```
	>> import neuralnet
	```

	"Creating an instance/ object of the NeuralNet class"
	```
	>> net = neuralnet.NeuralNet([784, 30, 10])
	```
	
	"Training and Testing the Neural Network to show results"
	```
	>> net.train(training_set = training_data, no_of_epochs = 30, size_of_batch = 10, learning_rate = 3.0, test_set = test_data)
	```

---

Feel free to play around with the parameters of the train function above.

Changing the structure of the neural network is also possible by increasing or decreasing the number of neurons in the hidden layer and also the number of hidden layers too, by changing parameters while creating a NeuralNet instance/ objects.