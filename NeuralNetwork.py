import math
import numpy as np

sigmoid = lambda x: 1/(1+math.exp(-x))
dsigmoid = lambda y: y * (1-y)

tanh = lambda x: math.tanh(x)
dtanh = lambda y: 1 - (y * y)

class NeuralNetwork:
    '''
    Perceptron
    '''
    def __init__(self, node_array, learning_rate, func, dfunc):
        
        # a list o all weights matrices
        self.weights = []
        # a list o all biases arrays
        self.biases = []
        # set the learning rate
        self.set_learning_rate(learning_rate)
        # set the activation functions
        self.set_activation_function(func, dfunc)
        # create the layer and bias matrices
        for i in range(1,len(node_array)):
            self.weights.append(np.random.rand(node_array[i], node_array[i-1]))
            self.biases.append(np.random.rand(node_array[i], 1))

    def set_learning_rate(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def set_activation_function(self, func=sigmoid, dfunc=dsigmoid):
        self.func = np.vectorize(func)
        self.dfunc = np.vectorize(dfunc)

    def feedfoward(self, array_input):        
        weights = []
        # transform array_input into a numpy array
        input = np.array(array_input).reshape(len(array_input),1)
        # calculate the weights of the first layer with input
        weights.append(self.weights[0] @ input)
        weights[0] += self.biases[0]
        weights[0] = self.func(weights[0])
        # calculate the weights
        for i in range(1,len(self.weights)):
            weights.append(self.weights[i] @ weights[i-1])
            weights[i] += self.biases[i]
            weights[i] = self.func(weights[i])
        # return the output
        return weights[-1].tolist()

    def train(self, array_input, array_target):
        # transform array_target and array_input into a numpy array
        target = np.array(array_target).reshape(len(array_target),1)
        input = np.array(array_input).reshape(len(array_input),1)
        ## Feedforward
        weights = []
        # transform array_input into a numpy array
        input = np.array(array_input).reshape(len(array_input),1)
        # recalculate the weights of the first layer with input
        weights.append(self.weights[0] @ input)
        weights[0] += self.biases[0]
        weights[0] = self.func(weights[0])
        # calculate the weights
        for i in range(1,len(self.weights)):
            weights.append(self.weights[i] @ weights[i-1])
            weights[i] += self.biases[i]
            weights[i] = self.func(weights[i])
        ##
        # calculate the output layer output's error and adjust it's weights
        errors = []
        errors.append(target - weights[-1])
        gradient = self.dfunc(weights[-1])
        gradient *= errors[0]
        gradient *= self.learning_rate

        transpose = np.transpose(weights[0])
        delta = gradient @ transpose

        self.weights[-1] += delta
        self.biases[-1] += gradient
        # calculate all layers errors and adjust the weights
        j=0
        for i in range(len(self.weights)-2,-1,-1):
            t1 = np.transpose(self.weights[i+1])
            errors.append(t1 @ errors[j])
            j+=1
            gradient = self.dfunc(weights[i])
            gradient *=  errors[j]
            gradient *= self.learning_rate
            if i != 0:
                transpose = np.transpose(self.weights[i-1])
            else:
                transpose = np.transpose(input)
            delta = gradient @ transpose
            self.weights[i] += delta
            self.biases[i] += gradient
            

