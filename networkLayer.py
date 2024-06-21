'''
This script will hold the class for individual layers of a neural network. 
They can be hidden layers or the 
output layer.


It should be able to:

Define a layer with a number of input and output nodes.
Generate all of the connections between the two sets of nodes (initially random).
Generate the biases for each of the output nodes.

Given some input data, calculate the weighted inputs based on the weights and biases.
Compute activation functions on a set of weighted inputs and output the result.

Created: 18/06/24
Last Update: 18/06/24
'''

# Import libraries needed
import numpy as np

# Dense layer class
class NetworkLayer:
    def __init__(self, numInputs, numOutputs):
        self.numInputNodes = numInputs
        self.numOutputNodes = numOutputs

        # All arrays for weights will have the input node on the first axis and the output node on the second
        self.weights = np.random.randn(numInputs, numOutputs)
        # Activations, weighted inputs, and biases will all be row vectors for dot products
        self.biases = np.zeros((1, numOutputs))
        # self.activations = np.zeros((1, numOutputs))

        # For training the network we will need to store the gradient of cost with respect to the weights and 
        # biases and so we create a place fo rthem on each layer
        self.costWeightGradients = np.zeros((numInputs, numOutputs))
        self.costBiasGradients = np.zeros((1, numOutputs))

    # Function that takes in the activations of the layer to the left and calculates the weighted inputs for its 
    # own activation functions. This can be sped up by matrix methods rather than doing each one individually
    def calculateWeightedInputs(self, prevActivations):
        return np.dot(prevActivations, self.weights) + self.biases
         
    # Function to compute a sigmoid activation based on weighted inputs. 1/(1 + e^-z)
    def sigmoidActivation(self, weightedInputs):
        sigmoidDenominator = np.exp(-weightedInputs) + 1
        self.activations = np.divide(1, sigmoidDenominator)

    # Function to compute a softmax activation based on weighted inputs
    def softmaxActivation(self, weightedInputs):
        # Compute the softmax function for the weighted inputs. e^zi / sum(e^zj)
        # We subtract the maximum to contol how large the exponential can grow (maximum 1)
        exponentialInputs = np.exp(weightedInputs - np.max(weightedInputs))
        sumOfExponentialInputs = np.sum(exponentialInputs)
        self.activations = np.divide(exponentialInputs, sumOfExponentialInputs)

    # A number of partial derivatives will be needed. These can be combined with one another to cancel terms. 
    # Once cancelled, each layer will have a set of derivative products that will be used repeatedly. these will 
    # be called the nodeValues. The nonde values are computed here, the final layer is distinct from the hidden 
    # layers.
    def calculateNodeValuesFinalLayer(self, yTrue):
        # convert yTrue to a np array
        yTrue = np.array([yTrue])

        costWrtSoftmax = np.divide(-1*yTrue, np.dot(yTrue, self.activations.T))
        # The elements needed for the softmax derivative matrix are calculated individually here.
        # diagonal elements are mi(1-mi) and off diagonal elements are -mi*mk
        activationSize = self.activations.shape[1]
        softmaxWrtWeightedInputs = np.zeros((activationSize, activationSize))

        for i in range(activationSize):
            activationI = self.activations[0, i]
            for k in range(activationSize):
                activationK = self.activations[0, k]
                if i == k:
                    softmaxWrtWeightedInputs[i, k] = activationI * (1 - activationI)
                else:
                    softmaxWrtWeightedInputs[i, k] = -activationI * activationK
        
        self.nodeValues = np.dot(costWrtSoftmax, softmaxWrtWeightedInputs)

    # For a hidden layer, we take the node values of the layer to the right and multiply them by 
    # dz/da_h * da_h/dz_h. dz/da_h = the weights in the the layer to the right
    def calculateNodeValuesHiddenLayer(self, rightLayer):
        # Each node in the current layer has multiple connections to the right layer and so we sum the
        # possible products. 
        rightLayerWeightsTranspose = np.transpose(rightLayer.weights)
        # Multiply this by the node values of the layer to the right.
        tempNodeValues = np.dot(rightLayer.nodeValues, rightLayerWeightsTranspose)

        # We find da_h/dz_h. This is a(1-a)
        activationsDerivative = np.multiply(self.activations, 1 - self.activations)

        # We multiply these two together to create the hidden layer's node values
        self.nodeValues = np.multiply(activationsDerivative, tempNodeValues)

    # We can calculate how the cost of the netwrok responds to each weight by finding dc/dw. To do so we 
    # take the node values and multiply them by dz/dw to give us dc/dw. Esentially just multiplying the node 
    # values of a layer by the activations of the layer to the left.
    # We will do so for a number of inputs in a batch and then average and so the results will be 
    # consecutively summed
    def calculatePartialCostGradients(self, leftLayerActivations):
        # To generate a matrix of the same shape as the weights, we need to transpose the left layer's 
        # activations
        leftLayerActivations = np.atleast_2d(leftLayerActivations).T
        weightGradients = np.dot(leftLayerActivations, self.nodeValues)

        # Add the gradients found from this input to the running total of the batch
        self.costWeightGradients += weightGradients

        # We do the same with the biases. These gradients are found by multiplying the node values by dz/db = 1
        self.costBiasGradients += self.nodeValues
