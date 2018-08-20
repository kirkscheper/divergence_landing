
import numpy as np
import random
import csv
import warnings

#warnings.simplefilter('error')

########################## Tools ##########################
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def relu(x):
    return np.maximum(x, 0.)

# generate random value in range [-w - 0.1, 2w + 0.1]
def perturb(x):
    return (3.*random.random() - 1.)*x + (2.*random.random() - 1.)*0.1

def load_nn(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')

        # first row contains the nn type
        type = reader.next()[0]

    if type == 'nn':
        network = nn()
    elif type == 'rnn':
        network = rnn()
    elif type == 'ctrnn':
        network = ctrnn()
    else:
        raise AssertionError('%s is not a known neural network type', type)

    network.load_weights(filename)
    return network

########################## NN ##########################
class nn(object):
    # generate array of floats that represent nerual weights
    def __init__(self, shape=[2,8,1]):
        self.nn_shape = shape

        # initialise weights including the bias neurons
        self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])

        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr] = 2*np.random.rand(self.nn_shape[layer_nr], self.nn_shape[layer_nr+1]) - 1.  # uniform distribution [-1,1]
            self.bias[layer_nr] = 2*np.random.rand(self.nn_shape[layer_nr+1]) - 1.                              # uniform distribution [-1,1]

    def reset(self):
        # nothing to see here
        return

    def predict(self, inputs, dt):
        activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])
        # set input activation
        activation[0][:] = np.array(inputs[:self.nn_shape[0]])

        # Feeforward network
        for layer_nr, weights_layer in enumerate(self.weights):
            activation[layer_nr+1][:] = activation[layer_nr].dot(weights_layer) + self.bias[layer_nr]

            # apply relu activation function to hidden layers
            if layer_nr < len(self.weights)-1:
                activation[layer_nr+1][:] = relu(activation[layer_nr+1])

        return activation[-1]

    def save_weights(self, filename, overwrite=False):
        with open(filename, 'w') as fp:
            np.savetxt(fp, ['nn'], fmt="%s")
            np.savetxt(fp, [self.nn_shape], delimiter=' ', fmt='%d')
            for w in self.weights:
                np.savetxt(fp, w, delimiter=' ')
            for b in self.bias:
                np.savetxt(fp, b, delimiter=' ')

    def load_weights(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            # first row contains the nn type
            type = reader.next()[0]
            if type != 'nn':
                raise AssertionError('The requested file is not a nn')

            # second row contains the size of the nn
            self.nn_shape = [int(i) for i in next(reader)]  # input, hidden layers, output layer

            self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])

            layer = 0
            neuron = 0
            parameter = 0   # 0 is wieght, 1 is bias
            for row in reader:
                if parameter == 0:
                    self.weights[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.weights):
                        layer = 0
                        parameter += 1
                else:
                    self.bias[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer+1]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.bias):
                        layer = 0
                        parameter += 1

    def mutate(self, mutation_rate=1.):
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            for neuron in xrange(self.nn_shape[layer_nr]):
                for connection in xrange(self.nn_shape[layer_nr+1]):
                    if random.random() <= mutation_rate:
                        self.weights[layer_nr][neuron][connection] = perturb(self.weights[layer_nr][neuron][connection])
                    if neuron == 0 and random.random() <= mutation_rate:
                        self.bias[layer_nr][connection] = perturb(self.bias[layer_nr][connection])

########################## RNN ##########################
class rnn(nn):
    # generate array of floats that represent nerual weights
    def __init__(self, shape=[2,8,1]):
        self.nn_shape = shape

        # define weight and persistent activations
        self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])
        self.recurrent_weights = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])

        self.prev_activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])

        # initialise weights including the bias and recurrent weights
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr] = 2*np.random.rand(self.nn_shape[layer_nr], self.nn_shape[layer_nr+1]) - 1.  # uniform distribution [-1,1]
            self.bias[layer_nr] = 2*np.random.rand(self.nn_shape[layer_nr+1]) - 1.                              # uniform distribution [-1,1]
            self.recurrent_weights[layer_nr] = 2*np.random.rand(self.nn_shape[layer_nr+1]) - 1.                 # uniform distribution [-1,1]

    def reset(self):
        self.prev_activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])

        # try to initialize network with zero inputs
        for _ in xrange(20):
            self.predict(np.zeros([self.nn_shape[0]]), 0.05)

    # dt should be in s
    def predict(self, inputs, dt):
        # activations include bias
        activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])

        # assign inputs
        activation[0][:] = np.array(inputs[:self.nn_shape[0]])

         # compute new activation
        for layer_nr, weights_layer in enumerate(self.weights):
            # compute feedforward activation
            activation[layer_nr+1][:] = activation[layer_nr].dot(weights_layer) + self.bias[layer_nr]

            # add recurrent activation
            activation[layer_nr+1] += self.recurrent_weights[layer_nr]*self.prev_activation[layer_nr+1]# * 1e3 / dt

            # apply relu activation function to hidden layers
            if layer_nr < len(self.weights)-1:
                activation[layer_nr+1][:] = relu(activation[layer_nr+1])

        self.prev_activation = np.copy(activation)

        return activation[-1]

    def save_weights(self, filename, overwrite=False):
        with open(filename, 'w') as fp:
            np.savetxt(fp, ["rnn"], fmt="%s")
            np.savetxt(fp, [self.nn_shape], delimiter=' ', fmt='%d')
            for w in self.weights:
                np.savetxt(fp, w, delimiter=' ')
            for b in self.bias:
                np.savetxt(fp, b, delimiter=' ')
            for r in self.recurrent_weights:
                np.savetxt(fp, r, delimiter=' ')

    def load_weights(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            # first row contains the nn type
            type = reader.next()[0]
            if type != 'rnn':
                raise AssertionError('The requested file is not a rnn')

            # second row contains the size of the nn
            self.nn_shape = [int(i) for i in next(reader)]  # input, hidden layers, output layer

            self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])
            self.recurrent_weights = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])

            layer = 0
            neuron = 0
            parameter = 0   # 0 is wieght, 1 is bias, 2 is recurrent weight
            for row in reader:
                if parameter == 0:
                    self.weights[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.weights):
                        layer = 0
                        parameter += 1
                elif parameter == 1:
                    self.bias[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer+1]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.bias):
                        layer = 0
                        parameter += 1
                else:
                    self.recurrent_weights[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer+1]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.recurrent_weights):
                        layer = 0
                        parameter += 1

    def mutate(self, mutation_rate=1.):
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            for neuron in xrange(self.nn_shape[layer_nr]):
                # mutate value in range [-w - 0.1, 2w + 0.1]
                for connection in xrange(self.nn_shape[layer_nr+1]):
                    if random.random() <= mutation_rate:
                        self.weights[layer_nr][neuron][connection] = perturb(self.weights[layer_nr][neuron][connection])
                    if neuron == 0:
                        if random.random() <= mutation_rate:
                            self.bias[layer_nr][connection] = perturb(self.bias[layer_nr][connection])
                        if random.random() <= mutation_rate:
                            self.recurrent_weights[layer_nr][connection] = perturb(self.recurrent_weights[layer_nr][connection])

########################## CTRNN ##########################
class ctrnn(rnn):
    # generate array of floats that represent nerual weights
    def __init__(self, shape=[2,8,1]):
        self.nn_shape = shape

        # define weight and persistent activations
        self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[:-1]])
        self.recurrent_weights = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])
        self.time_const = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        self.prev_activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])

        # initialise weights including the bias and recurrent weights
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr][:] = 2*np.random.rand(self.nn_shape[layer_nr], self.nn_shape[layer_nr+1]) - 1.  # uniform distribution [-1,1]
            self.bias[layer_nr][:] = 2*np.random.rand(self.nn_shape[layer_nr]) - 1.                                # uniform distribution [-1,1]
            self.recurrent_weights[layer_nr][:] = 2*np.random.rand(self.nn_shape[layer_nr+1]) - 1.                 # uniform distribution [-1,1]
            self.time_const[layer_nr][:] = np.random.rand(self.nn_shape[layer_nr])                                 # uniform distribution [ 0,1]

        self.time_const[-1][:] = np.random.rand(self.nn_shape[-1])                                                 # uniform distribution [ 0,1]

    def predict(self, inputs, dt):
        activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])

        # incorporate new inputs
        derviative = (np.array(inputs[:self.nn_shape[0]]) - self.prev_activation[0]) / (self.time_const[0] + dt)
        activation[0][:] = self.prev_activation[0] + dt*derviative

        # compute new activation
        for layer_nr, weights_layer in enumerate(self.weights):
            derviative = (sigmoid(activation[layer_nr] + self.bias[layer_nr]).dot(weights_layer) - self.prev_activation[layer_nr+1]) / (self.time_const[layer_nr+1] + dt)
            activation[layer_nr+1][:] = self.prev_activation[layer_nr+1] + dt*derviative

        self.prev_activation = np.copy(activation)

        #print 'activation', activation
        #print
        #quit(0)
        return activation[-1]

    def save_weights(self, filename, overwrite=False):
        with open(filename, 'w') as fp:
            np.savetxt(fp, ['ctrnn'], fmt="%s")
            np.savetxt(fp, [self.nn_shape], delimiter=' ', fmt='%d')
            for w in self.weights:
                np.savetxt(fp, w, delimiter=' ')
            for b in self.bias:
                np.savetxt(fp, b, delimiter=' ')
            for r in self.recurrent_weights:
                np.savetxt(fp, r, delimiter=' ')
            for t in self.time_const:
                np.savetxt(fp, t, delimiter=' ')

    def load_weights(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            # first row contains the nn type
            type = reader.next()[0]
            if type != 'ctrnn':
                raise AssertionError('The requested file is not a ctrnn')

            # second row contains the size of the nn
            self.nn_shape = [int(i) for i in next(reader)]  # input, hidden layers, output layer

            self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[:-1]])
            self.recurrent_weights = np.array([np.zeros(neurons) for neurons in self.nn_shape[1:]])
            self.time_const = np.array([np.zeros(neurons) for neurons in self.nn_shape])

            layer = 0
            neuron = 0
            parameter = 0   # 0 is wieght, 1 is bias, 2 is recurrent weight, 3 is time constant
            for row in reader:
                if parameter == 0:
                    self.weights[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.weights):
                        layer = 0
                        parameter += 1
                elif parameter == 1:
                    self.bias[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.bias):
                        layer = 0
                        parameter += 1
                elif parameter == 2:
                    self.recurrent_weights[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer+1]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.recurrent_weights):
                        layer = 0
                        parameter += 1
                else:
                    self.time_const[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                    if layer == len(self.time_const):
                        layer = 0
                        parameter += 1

    def mutate(self, mutation_rate=1.):
        # mutate value in range [-w - 0.1, 2w + 0.1]
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            for neuron in xrange(self.nn_shape[layer_nr]):
                if random.random() <= mutation_rate:
                    self.bias[layer_nr][neuron] = perturb(self.bias[layer_nr][neuron])
                if random.random() <= mutation_rate:
                    self.time_const[layer_nr][neuron] = perturb(self.time_const[layer_nr][neuron])
                    if self.time_const[layer_nr][neuron] < 0.:
                        self.time_const[layer_nr][neuron] = 0.

                for connection in xrange(self.nn_shape[layer_nr+1]):
                    if random.random() <= mutation_rate:
                        self.weights[layer_nr][neuron][connection] = perturb(self.weights[layer_nr][neuron][connection])
                    if neuron == 0:
                        if random.random() <= mutation_rate:
                            self.recurrent_weights[layer_nr][connection] = perturb(self.recurrent_weights[layer_nr][connection])

