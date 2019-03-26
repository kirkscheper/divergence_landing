
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

# generate random value in range [-w - 0.05, 2w + 0.05]
def perturb(x):
    #return x + (2.*random.random() - 1.)*x + (2.*random.random() - 1.)*0.05
    return (3.*random.random() - 1.)*x + (2.*random.random() - 1.)*0.05

# generate random value in range [-0.05, 2w + 0.05]
def perturb_u(x):
    val = 2.*random.random()*x + random.random()*0.05
    return val

# generate random value in range [-0.2,0.2]
def rand_w_init_s(*args, **kwargs):
    return (np.random.rand(*args, **kwargs) - .5) / 2.5  # uniform distribution 


def load_nn(filename, print_weights=False):
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

    network.load_weights(filename, print_weights)
    return network

########################## NN ##########################
class nn(object):
    # generate array of floats that represent nerual weights
    def __init__(self, shape=[2,8,1], act_func=relu):
        self.nn_shape = shape
        self.act_func = act_func

        # initialise weights including the bias neurons
        self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        for layer_nr, layer_shape in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr] = rand_w_init_s(layer_shape, self.nn_shape[layer_nr+1])
        for layer_nr, layer_shape in enumerate(self.nn_shape):
            self.bias[layer_nr] = rand_w_init_s(layer_shape)

    def reset(self):
        # nothing to see here
        return

    def predict(self, inputs, t):
        activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])

        # set input activation
        activation[0][:] = np.array(inputs[:self.nn_shape[0]]) + self.bias[0]

        # Feeforward network
        for layer_nr, weights_layer in enumerate(self.weights):
            activation[layer_nr+1][:] = activation[layer_nr].dot(weights_layer) + self.bias[layer_nr+1]

            # apply relu activation function to hidden layers
            if layer_nr + 1 < len(self.weights):
                activation[layer_nr+1][:] = self.act_func(activation[layer_nr+1])

        return activation[-1]

    def save_weights(self, filename, overwrite=False):
        with open(filename, 'w') as fp:
            np.savetxt(fp, ['nn'], fmt="%s")
            np.savetxt(fp, [self.nn_shape], delimiter=' ', fmt='%d')
            for w in self.weights:
                np.savetxt(fp, w, delimiter=' ')
            for b in self.bias:
                np.savetxt(fp, b, delimiter=' ')

    def load_weights(self, filename, print_weights=True):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            # first row contains the nn type
            type = reader.next()[0]
            if type != 'nn':
                raise AssertionError('The requested file is not a nn')

            # second row contains the size of the nn
            self.nn_shape = [int(i) for i in next(reader)]  # input, hidden layers, output layer

            self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape])

            layer = 0
            neuron = 0
            parameter = 0   # 0 is wieght, 1 is bias
            for row in reader:
                if parameter == 0:
                    self.weights[layer][neuron] = np.array(row)
                    if print_weights: print '{'+ "".join("%.6f," % float(x) for x in row) + '},'
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                        if print_weights: print
                    if layer == len(self.weights):
                        layer = 0
                        parameter += 1
                        if print_weights: print
                        if print_weights: print
                else:
                    self.bias[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        if print_weights: print '{'+ "".join("%.6f," % float(x) for x in self.bias[layer]) + '};'
                        neuron = 0
                        layer += 1
                    if layer == len(self.bias):
                        layer = 0
                        parameter += 1
                        if print_weights: print

    def mutate(self, mutation_rate=1.):
        for layer_nr, s in enumerate(self.nn_shape):
            for neuron in xrange(s):
                if random.random() <= mutation_rate:
                    self.bias[layer_nr][neuron] = perturb(self.bias[layer_nr][neuron])
                if layer_nr + 1 < len(self.nn_shape):
                    for connection in xrange(self.nn_shape[layer_nr+1]):
                        if random.random() <= mutation_rate:
                            self.weights[layer_nr][neuron][connection] = perturb(self.weights[layer_nr][neuron][connection])

########################## RNN ##########################
class rnn(nn):
    # generate array of floats that represent nerual weights
    def __init__(self, shape=[2,8,1], act_func=relu):
        self.nn_shape = shape
        self.act_func = act_func

        # define weight and persistent activations
        self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape])
        self.recurrent_weights = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        self.prev_activation = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        # initialise weights including the bias and recurrent weights
        for layer_nr, layer_shape in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr] = rand_w_init_s(layer_shape, self.nn_shape[layer_nr+1])
        for layer_nr, layer_shape in enumerate(self.nn_shape):
            self.bias[layer_nr] = rand_w_init_s(layer_shape)
            self.recurrent_weights[layer_nr] = rand_w_init_s(layer_shape)

    def reset(self):
        self.prev_activation.fill(0)

        # try to initialize network with zero inputs
        for _ in xrange(20):
            self.predict(np.zeros([self.nn_shape[0]]), 0.)

    # t should be in s
    def predict(self, inputs, t):
        # activations include bias
        activation = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        # assign inputs
        activation[0][:] = np.array(inputs[:self.nn_shape[0]]) + self.bias[0] + self.recurrent_weights[0]*self.prev_activation[0]
        
         # compute new activation
        for layer_nr, weights_layer in enumerate(self.weights):
            # compute activation
            activation[layer_nr+1][:] = activation[layer_nr].dot(weights_layer) + self.bias[layer_nr+1] + self.recurrent_weights[layer_nr+1]*self.prev_activation[layer_nr+1]

            # apply relu activation function to hidden layers
            if layer_nr + 1 < len(self.weights):
                activation[layer_nr+1][:] = self.act_func(activation[layer_nr+1])

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

    def load_weights(self, filename, print_weights=False):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            # first row contains the nn type
            type = reader.next()[0]
            if type != 'rnn':
                raise AssertionError('The requested file is not a rnn')

            # second row contains the size of the nn
            self.nn_shape = [int(i) for i in next(reader)]  # input, hidden layers, output layer

            self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape])
            self.recurrent_weights = np.array([np.zeros(neurons) for neurons in self.nn_shape])

            layer = 0
            neuron = 0
            parameter = 0   # 0 is wieght, 1 is bias, 2 is recurrent weight
            for row in reader:
                if parameter == 0:
                    self.weights[layer][neuron] = np.array(row)
                    if print_weights: print '{'+ "".join("%.6f," % float(x) for x in row) + '},'
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                        if print_weights: print
                    if layer == len(self.weights):
                        layer = 0
                        parameter += 1
                        if print_weights: print
                        if print_weights: print
                elif parameter == 1:
                    self.bias[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        if print_weights: print '{'+ "".join("%.6f," % float(x) for x in self.bias[layer]) + '};'
                        neuron = 0
                        layer += 1
                    if layer == len(self.bias):
                        layer = 0
                        parameter += 1
                        if print_weights: print
                        if print_weights: print
                else:
                    self.recurrent_weights[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        if print_weights: print '{'+ "".join("%.6f," % float(x) for x in self.recurrent_weights[layer]) + '},'
                        neuron = 0
                        layer += 1
                    if layer == len(self.recurrent_weights):
                        layer = 0
                        parameter += 1
                        if print_weights: print

    def mutate(self, mutation_rate=1.):
        for layer_nr, s in enumerate(self.nn_shape):
            for neuron in xrange(s):
                if random.random() <= mutation_rate:
                    self.bias[layer_nr][neuron] = perturb(self.bias[layer_nr][neuron])
                if random.random() <= mutation_rate:
                    self.recurrent_weights[layer_nr][neuron] = perturb(self.recurrent_weights[layer_nr][neuron])
                if (layer_nr + 1 < len(self.nn_shape)):
                    for connection in xrange(self.nn_shape[layer_nr+1]):
                        if random.random() <= mutation_rate:
                            # mutate value in range [-w - 0.1, 2w + 0.1]
                            self.weights[layer_nr][neuron][connection] = perturb(self.weights[layer_nr][neuron][connection])

########################## CTRNN ##########################
class ctrnn(rnn):
    # generate array of floats that represent nerual weights
    def __init__(self, shape=[2,8,1], act_func=np.tanh):
        self.nn_shape = shape
        self.act_func = act_func

        self.last_t = 0.

        # define weight and persistent activations
        self.weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape[:-1]])
        self.time_const = np.array([np.zeros(neurons) for neurons in self.nn_shape])
        self.gain = np.array([np.zeros(neurons) for neurons in self.nn_shape[:-1]])
        
        print np.shape(self.bias)

        self.prev_activation = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        # initialise weights including the bias and recurrent weights
        for layer_nr, layer_shape in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr][:] = rand_w_init_s()
            self.bias[layer_nr][:] = rand_w_init_s()
            self.gain[layer_nr][:] = np.random.rand(layer_shape)                          # uniform distribution [ 0,1]
        for layer_nr, layer_shape in enumerate(self.nn_shape):
            self.time_const[layer_nr][:] = np.random.rand(layer_shape)                    # uniform distribution [ 0,1]

    def predict(self, inputs, t):
        activation = np.array([np.zeros(neurons) for neurons in self.nn_shape])
        
        if self.last_t < t:
            dt = t - self.last_t
        else:
            dt = 0.025

        self.last_t = t

        # incorporate new inputs
        derviative = (np.array(inputs[:self.nn_shape[0]]) - self.prev_activation[0]) / (self.time_const[0] + dt)
        activation[0][:] = self.prev_activation[0] + dt*derviative
        
        # compute new activation
        for layer_nr, weights_layer in enumerate(self.weights):
            derviative = (self.act_func(self.gain[layer_nr]*(activation[layer_nr] + self.bias[layer_nr])).dot(weights_layer) - self.prev_activation[layer_nr+1]) / (self.time_const[layer_nr+1] + dt)
            activation[layer_nr+1][:] = self.prev_activation[layer_nr+1] + dt*derviative

        self.prev_activation = np.copy(activation)

        return activation[-1]

    def save_weights(self, filename, overwrite=False):
        with open(filename, 'w') as fp:
            np.savetxt(fp, ['ctrnn'], fmt="%s")
            np.savetxt(fp, [self.nn_shape], delimiter=' ', fmt='%d')
            for w in self.weights:
                np.savetxt(fp, w, delimiter=' ')
            for b in self.bias:
                np.savetxt(fp, b, delimiter=' ')
            for t in self.time_const:
                np.savetxt(fp, t, delimiter=' ')
            for g in self.gain:
                np.savetxt(fp, g, delimiter=' ')

    def load_weights(self, filename, print_weights=False):
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
            self.time_const = np.array([np.zeros(neurons) for neurons in self.nn_shape])
            self.gain = np.array([np.zeros(neurons) for neurons in self.nn_shape[:-1]])

            layer = 0
            neuron = 0
            parameter = 0   # 0 is wieght, 1 is bias, 2 is recurrent weight, 3 is time constant
            for row in reader:
                if parameter == 0:
                    self.weights[layer][neuron] = np.array(row)
                    if print_weights: print '{'+ "".join("%.6f," % float(x) for x in row) + '},'
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        neuron = 0
                        layer += 1
                        if print_weights: print
                    if layer == len(self.weights):
                        layer = 0
                        parameter += 1
                        if print_weights: print
                        if print_weights: print
                elif parameter == 1:
                    self.bias[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        if print_weights: print '{'+ "".join("%.6f," % float(x) for x in self.bias[layer]) + '};'
                        neuron = 0
                        layer += 1
                    if layer == len(self.bias):
                        layer = 0
                        parameter += 1
                        if print_weights: print
                        if print_weights: print
                elif parameter == 2:
                    self.time_const[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        if print_weights: print '{'+ "".join("%.6f," % float(x) for x in self.time_const[layer]) + '};'
                        neuron = 0
                        layer += 1
                    if layer == len(self.time_const):
                        layer = 0
                        parameter += 1
                        if print_weights: print
                elif parameter == 3:
                    self.gain[layer][neuron] = np.array(row)
                    neuron += 1
                    if neuron == self.nn_shape[layer]:
                        if print_weights: print '{'+ "".join("%.6f," % float(x) for x in self.gain[layer]) + '};'
                        neuron = 0
                        layer += 1
                    if layer == len(self.gain):
                        layer = 0
                        parameter += 1
                        if print_weights: print

    def mutate(self, mutation_rate=1.):
        # mutate value in range [-w - 0.1, 2w + 0.1]
        for layer_nr, _ in enumerate(self.nn_shape):
            for neuron in xrange(self.nn_shape[layer_nr]):
                if random.random() <= mutation_rate:
                    self.time_const[layer_nr][neuron] = perturb_u(self.time_const[layer_nr][neuron])
                if layer_nr + 1 < len(self.nn_shape):
                    if random.random() <= mutation_rate:
                        self.bias[layer_nr][neuron] = perturb(self.bias[layer_nr][neuron])
                    if random.random() <= mutation_rate:
                        self.gain[layer_nr][neuron] = perturb_u(self.gain[layer_nr][neuron])
                    for connection in xrange(self.nn_shape[layer_nr+1]):
                        if random.random() <= mutation_rate:
                            self.weights[layer_nr][neuron][connection] = perturb(self.weights[layer_nr][neuron][connection])


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hello.')
    parser.add_argument("-a", "--agent", type=str, default='', help="The full path to the desired agent")
    args = parser.parse_args()

    agent = load_nn(args.agent, print_weights=True)
    for i in xrange(100):
        print agent.predict(np.array([1., 1.]), 0.)

