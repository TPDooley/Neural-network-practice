'''
This script will serve as the training data, and function, that we wish the neural network to solve.

It should be able to:

Generate a large set of data based on some function. The data will be coordinates followed by a one-hot vector 
characterisation. (3D array)

Take a single point in 2D space and catagorise it as either within or outside the accepted region, returning 
a one-hot vector of the results.

Created: 18/06/24
Last Update: 18/06/24
'''
# Import libraries needed
import random as rd

# Superclass of the data set to create it and populate it with data
class dataSet:
    def __init__(self):
        self.dataPoints = []
    
    # Generate a large data set held by the object
    def generateData(self, numPoints):
        for i in range(numPoints):
            # Generate a point
            x = rd.random()
            y = rd.random()
            # Calculate the desired output for the network
            oneHot = self.functionCallCheck(x, y)
            # The data added to the set's array is the coordinate in question, followed by 
            # one-hot vector that caracterises that point's output
            self.dataPoints.append([[x, y], oneHot])

    # Generate a batch for the network to train on
    def generateBatch(self, batchSize):
        outputBatch = []

        for i in range(batchSize):
            # Generate a point
            x = rd.uniform(0, 1)
            y = rd.uniform(0, 1)

            # Calculate the desired output for the network
            oneHot = self.functionCallCheck(x, y)

            # The data added to the set's array is the coordinate in question, followed by 
            # one-hot vector that caracterises that point's output
            outputBatch.append([[x, y], oneHot])
            
        return outputBatch


# Specific function that will be used to train the network on
class polynomialFunction(dataSet):
    # Function to take in a coordinate and output the network's desired result as a one-hot vector
    def functionCallCheck(self, x, y):
        # Calculate chosen function
        xFunc = x**5 + 3*x**4 + x**2 + 0.1
        # Determine correct one-hot vector to output
        if y > xFunc:
            oneHot = [1,0]
        else:
            oneHot = [0,1]
        return oneHot
