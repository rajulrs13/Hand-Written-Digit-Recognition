# Importing the libraries
import numpy as np
import random

# A class to create neural net instances/ objects.
class NeuralNet(object):

	"""
	Input argument 'layers' is an array containing
	structure of the network including output and
	input layers i.e how many neurons are present
	in the respective layers.
	"""

	# Constructor
	def __init__(self, layers):

		# No. of Layers in the Network
		self.num_of_layers = len(layers)

		# Layered Structure of the Network in form of array
		self.layers = layers

		# Setting Random Biases for all layers except Input Layer
		self.biases = []
		for layer in layers[1: ]:
			self.biases.append(np.random.randn(layer, 1))

		# Setting Random Weights for all Synapses, layer by layer
		self.weights = []
		for x, y in zip(layers[: -1], layers[1: ]):
			self.weights.append(np.random.randn(y, x))


	# Funtion to feed forward and compute the Output Layer
	def feed_forward(self, X):

		# Initialise Output
		y_hat = X

		# Computing Output Layer
		for bias, weight in zip(self.biases, self.weights):
			y_hat = sigmoid(np.dot(weight, y_hat) + bias)

		# Output Layer is returned
		return y_hat


	"""
	Function to train the Neural Network on Training Set
	using Stochastic Gradient Descent 
	"""
	def train(self, training_set, no_of_epochs, size_of_batch, learning_rate, test_set = None):
		
		# Length of Test Data (Optional)
		if test_set:
			test_set = list(test_set)
			len_of_test = len(test_set)

		# Length of the Training Set
		training_set = list(training_set)
		len_of_training_set = len(training_set)

		# For each epoch
		for epoch in range(no_of_epochs):

			# Shuffling the Training Set
			random.shuffle(training_set)

			# Making small batches out of the Training Set
			batches = []
			for j in range(0, len_of_training_set, size_of_batch):
				batches.append(training_set[j: j + size_of_batch])

			# Training each batch
			for batch in batches:
				self.train_batch(batch, learning_rate)

			"""
			If Test Data is provided then no of correct
			predictions is calculated in each epoch
			"""
			max_correct = -1
			if test_set:
				correct = self.predict(test_set)
				if correct > max_correct:
					max_correct = correct
				print("Epoch {}: Correct = {} / {}.".format(epoch, correct, len_of_test))
			else:
				print("Epoch {} completed".format(epoch))

		# Best Accuracy of the Model
		if test_set:
			print("The Best Accuracy of the model is {}%.".format(max_correct / len_of_test * 100))


	# Training function for each batch
	def train_batch(self, batch, learning_rate):
		
		# Length of the Batch
		len_of_batch = len(batch)

		# Initialising ∇biases
		del_biases = []
		for bias in self.biases:
			del_biases.append(np.zeros(bias.shape))

		# Initialising ∇weights
		del_weights = []
		for weight in self.weights:
			del_weights.append(np.zeros(weight.shape))

		# In each batch
		for X, y in batch:
			
			# Backpropagating to obtain change in ∇biases & ∇weights
			delta_del_biases, delta_del_weights = self.backpropagate(X, y)

			# Updating ∇biases
			del_bias = [del_bias + delta_del_bias for del_bias, delta_del_bias in zip(del_biases, delta_del_biases)]

			# Updating ∇weights
			del_weights = [del_weight + delta_del_weight for del_weight, delta_del_weight in zip(del_weights, delta_del_weights)]
		
		# Updating values of Biases
		self.biases = [bias - ((learning_rate * del_bias) / len_of_batch) for bias, del_bias in zip(self.biases, del_biases)]
		
		# Updating values of Weights
		self.weights = [weight - ((learning_rate * del_weight) / len_of_batch) for weight, del_weight in zip(self.weights, del_weights)]


	# Function for backpropagation
	def backpropagate(self, X, y):

		# Initialising ∇biases
		del_biases = []
		for bias in self.biases:
			del_biases.append(np.zeros(bias.shape))

		# Initialising ∇weights
		del_weights = []
		for weight in self.weights:
			del_weights.append(np.zeros(weight.shape))

		""" Forward Pass """
		# Intialising activation
		activation = X

		# List to store all activations, layer by layer
		list_of_activations = [activation]

		# List to store all z vectors, layer by layer
		list_of_z = []

		# Feed Forward Step
		for bias, weight in zip(self.biases, self.weights):
			z = np.dot(weight, activation) + bias
			list_of_z.append(z)
			activation = sigmoid(z)
			list_of_activations.append(activation)

		""" Backward Pass """
		# For last-layer (Initialisation)
		delta = (list_of_activations[-1] - y) * derivative_of_sigmoid(list_of_z[-1])
		del_biases[-1] = delta
		del_weights[-1] = np.dot(delta, list_of_activations[-2].transpose())

		# For second-last layer onwards
		for layer in range(2, self.num_of_layers):
			z = list_of_z[-layer]
			delta = np.dot(self.weights[-layer + 1].transpose(), delta) * derivative_of_sigmoid(z)
			del_biases[-layer] = delta
			del_weights[-layer] = np.dot(delta, list_of_activations[-layer - 1].transpose())

		# Return the final values of ∇biases and ∇weights as a tuple
		return (del_biases, del_weights)

	def predict(self, test_data):
		test_results = []
		for X, y in test_data:
			test_results.append((np.argmax(self.feed_forward(X)), y))
		correct = 0
		for X, y in test_results:
			if X == y:
				correct = correct + 1
		return correct


# Other Non-Class Methods

# Sigmoid Activation Function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# Derivative of Sigmoid Function
def derivative_of_sigmoid(z):
	return sigmoid(z) * (1 - sigmoid(z))





