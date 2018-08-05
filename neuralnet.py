# Importing the libraries
import numpy as np
import random

# A class to create neural net instances/ objects.
class neuralnet(object):

	"""
	Input arguement layers is an array 
	containing structure of the network
	including output and input layers 
	i.e how many neurons are present in
	the respective layers.
	"""

	# Constructor
	def __init__(self, layers):

		# No. of Layers in the Network
		self.num_of_layers = len(layers)

		# Layered Structure of the Network in form of array
		self.layers = layers

		# Setting Random Biases for all layers except Input Layer
		self.biases = []
		for i in layers[1: ]:
			self.biases.append(np.random.randn(i, 1))

		# Setting Random Weights for all Synapses, layer by layer
		self.weights = []
		for x, y in zip(layers[: -1], layers[1: ]):
			self.weights.append(np.random.randn(y, x))




		