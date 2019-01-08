import math
import numpy as np

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = np.random.rand(self.hidden_nodes,self.input_nodes)
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)

        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        self.bias_o = np.random.rand(self.output_nodes, 1)

        self.learning_rate = 0.1
        self.sigmoid = np.vectorize(lambda x: 1/(1+math.exp(-x)))
        self.dsigmoid = np.vectorize(lambda y: y * (1-y)) #y is already sigmoided

    def feedfoward(self, array_input):

        #generating hidden outputs
        inputs = np.array(array_input).reshape(len(array_input),1)
        hidden = self.weights_ih @ inputs
        hidden += self.bias_h

        #activation function
        hidden = self.sigmoid(hidden)

        #generating the output's output
        output = self.weights_ho @ hidden
        output+=self.bias_o
        output = self.sigmoid(output)

        return output.tolist()

    def train(self, array_input, array_target):
        '''
        first run a feedfoward;
        second calculate the output error;
        then calculate the hidden output error
        '''

        ###############################feedfoward#################
        #generating hidden outputs
        inputs = np.array(array_input).reshape(len(array_input),1)
        hidden = self.weights_ih @ inputs
        hidden += self.bias_h

        #activation function
        hidden = self.sigmoid(hidden)

        #generating the output's output
        outputs = self.weights_ho @ hidden
        outputs += self.bias_o
        outputs = self.sigmoid(outputs)

        ##########################################################
        ####################calculate the output error############

        targets = np.array(array_target).reshape(len(array_target),1)

        #calculate the errors
        output_errors = targets - outputs
        #calculate output gradients
        '''calculate the derivate of the sigmoid function for all the matrix elements'''
        output_gradient = self.dsigmoid(outputs)
        output_gradient*=output_errors
        output_gradient*=self.learning_rate
        #calculate hidden deltas
        hidden_T = np.transpose(hidden)
        weight_ho_delta = output_gradient @ hidden_T

        #adjust the weights by its deltas
        self.weights_ho+=weight_ho_delta
        #adjust the bias by its deltas(which is the gradients)
        self.bias_o+=output_gradient
        ##########################################################
        #############calculate the hidden output error############
        #calculate the errors in the weights of the hidden to output layers
        weights_ho_transpose = np.transpose(self.weights_ho)
        errors_ho = weights_ho_transpose @ output_errors

        #hidden gradient
        hidden_gradient = self.dsigmoid(hidden)
        hidden_gradient *= errors_ho
        hidden_gradient *=self.learning_rate

        #input delta
        input_T = np.transpose(inputs)
        weight_ih_delta = hidden_gradient @ input_T

        #adjust the weights by its delta
        self.weights_ih+=weight_ih_delta
        #adjust by the gradient
        self.bias_h+=hidden_gradient
        ##########################################################
