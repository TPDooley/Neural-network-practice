'''
This script will hold the main network and have functionality that concerns the whole network.


It should be able to:

Create a neural network by storing all of the layer objects initialised with the correct sizes.

Take in a data point (batch of size one), format it, and pass it through the layers to recieve an output.

Calculate the cost of an output based on the desired result (true value).

Take in data batch to train itself by passing the data forward and backwards (back propogation) before 
calling each layer to update their weights and biases.

Created: 18/06/24
Last Update: 18/06/24
'''

# Libraries needed
import numpy as np
# Custom network layer class
import networkLayer as nl 

# Create the network class
class NeuralNetwork:
    def __init__(self, layerSizes):
        self.layers = []
        for i in range(len(layerSizes) - 1):
            # Append a layer for each of the layer sizes, passing the number of input nodes and output nodes.
            # The first layer is just inputs and so we append one less.
            self.layers.append(nl.NetworkLayer(layerSizes[i], layerSizes[i+1]))

    # Function for passing a single set of input activations to see if the network can predict the outcome
    def calculateSinglePoint(self, data):
        # Input data passed as a list continaing the vector
        inputs = np.copy(data)
        return self.forwardPass(inputs)

    # def inputDataBatch(self, data):
    #     # All data should be inputted as a 2 or 3 dimensional list. 3D denotes data and an expected result.
    #     self.dataBatch = np.copy(data)

    # Function for taking a single set of input activations and passing them through the whole network and 
    # returning the output activations
    def forwardPass(self, inputs):
        for i in range(len(self.layers)):
            currentLayer = self.layers[i]
            # On each layer, take in inputs and calculate the weighted inputs using weights and biases
            weightedInputs = currentLayer.calculateWeightedInputs(inputs)

            # Now calculate the activations of the layer using the weighted inputs
            if i == (len(self.layers) - 1):
                # Final layer of the network and so use softmax activation
                currentLayer.softmaxActivation(weightedInputs)
                # currentLayer.sigmoidActivation(weightedInputs)
            else:
                # Not the final layer and so use sigmoid activation
                currentLayer.sigmoidActivation(weightedInputs)

            # Finally, update the 'inputs' to be the output of the last last layer -for the next layer
            inputs = currentLayer.activations
        # Take the activations of the final layer as the ouput of the network
        return self.layers[-1].activations

    # Function for taking in a batch and training the network on it
    def trainWeightsAndBiases(self, batch, learnRate):
        # The batch is a 3D python list, we pass each point one at a time
        for i in range(len(batch)):
            # Extract the input data and the desired outcome
            inputs = batch[i][0]
            yTrue = batch[i][1]

            # Pass the inputs through the network to compute all the activation layers
            self.forwardPass(inputs)

            # Now we compute the node values of each layer in reverse. 
            # This back propogation saves on calculations we can reuse.
            # This for loop traverses backwards providing both the layer index and the layer itself
            for index, layer in reversed(list(enumerate(self.layers))):
                # We calculate the node values.
                if index == len(self.layers) - 1:
                    # The final layer takes in the desired output found in the batch
                    layer.calculateNodeValuesFinalLayer(yTrue)
                else:
                    # Hiddenlayers take the layer to the right as an input
                    layer.calculateNodeValuesHiddenLayer(self.layers[index + 1])

                # With these calculations done we can now compute all of the cost gradients for each layer
                # The take the activations of the layer to the left as an input, the layer leftmost takes in 
                # the input data
                if index == 0:
                    # Left most layer, pass inputs
                    layer.calculatePartialCostGradients(inputs)
                else:
                    # Any other layer, pass left layer activations
                    leftLayerActivations = self.layers[index - 1].activations
                    layer.calculatePartialCostGradients(leftLayerActivations)

        # Now that the layers each have a matrix of gradient changes, summed up over the batch, we just 
        # need to average them and apply them to the weights and biases of each layer
        self.updateAllWeightsAndBiases(len(batch), learnRate)

    # Function to apply the found gradients to the weights and biases of all the layers
    def updateAllWeightsAndBiases(self, batchSize, learnRate):
        for layer in self.layers:
            # Average over the batch
            averageWeightGradient = np.divide(layer.costWeightGradients, batchSize)
            averageBiasGradient = np.divide(layer.costBiasGradients, batchSize)

            # Apply the gradient descent step
            layer.weights -= averageWeightGradient*learnRate
            layer.biases -= averageBiasGradient*learnRate
            # print("Weights: ", layer.weights)
            # print("Biases: ", layer.biases)

            # Finally, reset the gradient matrices for the next learning batch
            layer.costWeightGradients = np.zeros((layer.numInputNodes, layer.numOutputNodes))
            layer.costBiasGradients = np.zeros((1, layer.numOutputNodes))

# Take in a grid of points and test the network on every point
    def renderCurrentAccuracy(self, start, end, step):
        # colours = []
        # Generate a grid of values to test
        x = np.arange(start, end, step)
        y = np.arange(start, end, step)
        # Pass each point through the network

        outputPoints = []

        for i in range(len(x)):
            outputPoints.append(0)
            for j in range(len(y)):
                networkOutput = self.forwardPass([x[i], y[j]])
                if networkOutput[:, 0].round(0) == 0:
                    outputPoints[i] = y[j]
        
        return outputPoints
    


# Class for the cost fucntion (catagorical cross entropy)
class Cost:
    def forwardPass(self, network, batch):
        runningTotal = 0
        for i in range(len(batch)):
            inputs = batch[i][0]
            yTrue = batch[i][1]

            networkOuput = network.forwardPass(inputs)
            networkOuputClipped = np.clip(networkOuput, 1e-7, 1-1e-7)

            correctConfidences = np.sum(networkOuputClipped*yTrue, axis=1)

            # Compute the loss on the found value
            negLogLikelyhoods = -np.log(correctConfidences)
            # Return the array of loss calculations found from the batch
            runningTotal += negLogLikelyhoods
        # Return the average cost over the batch
        return runningTotal / len(batch)