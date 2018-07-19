
########################## NN ##########################
class nn(object):
    # generate array of floats that represent nerual weights
    def __init__(self):
        self.nn_shape = [2,8,1]  # input, hidden layers, output layer
        
        # initialise weights including the bias neurons
        self.weights = np.array([np.zeros([self.nn_shape[layer]+1, self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            for neuron in xrange(self.nn_shape[layer_nr]):
                for weight in xrange(self.nn_shape[layer_nr+1]):
                    # uniform distribution [-1,1]
                    self.weights[layer_nr][neuron][weight] = 2*random.random() - 1.
         
    def predict(self, inputs):
        activation = np.array([np.zeros(self.nn_shape[layer]+1) for layer, _ in enumerate(self.nn_shape)])
        # add bias to inputs
        activation[0][:-1] = np.array(inputs[:self.nn_shape[0]])
        for layer_nr, weights_layer in enumerate(self.weights):
            activation[layer_nr][-1] = 1 # set bias activation
            activation[layer_nr+1][:-1] = activation[layer_nr].dot(weights_layer)
            # apply relu activation function to hidden layers
            if layer_nr < len(self.weights)-1:
                np.maximum(activation[layer_nr+1], 0., activation[layer_nr+1])

        return activation[-1][:-1]  # ignore bias neuron in putput layer

    def save_weights(self, filename, overwrite=False):
        with open(filename, 'w') as fp:
            np.savetxt(fp, [self.nn_shape], delimiter=' ')
            for layer in self.weights:
                np.savetxt(fp, layer, delimiter=' ')

    def load_weights(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            
            # first row contains the size of the nn
            self.nn_shape = [int(float(i)) for i in next(reader)]  # input, hidden layers, output layer
            
            self.weights = np.array([np.zeros([self.nn_shape[layer]+1, self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            
            layer = 0
            neuron = 0
            weight = 0
            first_run = True
            for row in reader:
                if first_run:
                    first_run = False
                    continue
                for w in row:
                    if weight and weight % len(self.weights[layer][neuron]) is 0:
                        neuron += 1
                        weight = 0
                    if neuron and neuron % len(self.weights[layer]) is 0:
                        layer += 1
                        neuron = 0
                    self.weights[layer][neuron][weight] = w
                    weight += 1

    def mutSet(individual):
        """Mutation that pops or add an element."""
        for i in range(len(individual[0].weights)):
            for j in range(len(individual[0].weights[i])):
                if random.random() < MUTATION_RATE:
                    # mutate value in range [-w - 0.1, 2w + 0.1]
                    individual[0].weights[i][j] = (3*random.random() - 1)*individual[0].weights[i][j] + (2*random.random() - 1)*0.2
        return individual,

########################## RNN ##########################
class rnn(nn):
    # generate array of floats that represent nerual weights
    def __init__(self):
        self.nn_shape = [3,8,1]  # input, hidden layers, output layer
        
        # define weight and persistent activations
        self.weights = np.array([np.zeros([self.nn_shape[layer]+2, self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        self.prev_activation = np.array([np.zeros(self.nn_shape[layer]+1) for layer, _ in enumerate(self.nn_shape)])
        
        # initialise weights including the bias and recurrent weights
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            for neuron in xrange(self.nn_shape[layer_nr]+2):
                for weight in xrange(self.nn_shape[layer_nr+1]):
                    # uniform distribution [-1,1]
                    self.weights[layer_nr][neuron][weight] = 2*random.random() - 1.
         
    def predict(self, inputs):
        # activations include bias
        activation = np.array([np.zeros(self.nn_shape[layer]+1) for layer, _ in enumerate(self.nn_shape)])
        # assign inputs
        activation[0][:-1] = np.array(inputs[:self.nn_shape[0]])
        for layer_nr, weights_layer in enumerate(self.weights):
            activation[layer_nr][-1] = 1 # set bias activation
            activation[layer_nr+1][:-1] = activation[layer_nr].dot(weights_layer[:-1])

            # add recurrent path
            for neuron, w in enumerate(weights_layer[-1][:-1]):
                activation[layer_nr+1][neuron] += w*self.prev_activation[layer_nr+1][neuron]

            # apply relu activation function to hidden layers
            if layer_nr < len(self.weights)-1:
                np.maximum(activation[layer_nr+1], 0., activation[layer_nr+1])

        self.prev_activation = np.copy(activation)
        
        #print 'activation', activation
        
        return activation[-1][:-1]  # ignore bias neuron in output layer
        
    def load_weights(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            
            # first row contains the size of the nn
            self.nn_shape = [int(float(i)) for i in next(reader)]  # input, hidden layers, output layer
            
            # define weight and persistent activations
            self.weights = np.array([np.zeros([self.nn_shape[layer]+2, self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            
            self.prev_activation = np.array([np.zeros(self.nn_shape[layer]+1) for layer, _ in enumerate(self.nn_shape)])
            
            layer = 0
            neuron = 0
            weight = 0
            for row in reader:
                for w in row:
                    if weight and weight % len(self.weights[layer][neuron]) is 0:
                        neuron += 1
                        weight = 0
                    if neuron and neuron % len(self.weights[layer]+1) is 0:
                        layer += 1
                        neuron = 0
                    self.weights[layer][neuron][weight] = w
                    weight += 1

########################## CTRNN ##########################
class ctrnn(nn):
    # generate array of floats that represent nerual weights
    def __init__(self):
        self.nn_shape = [2,8,1]  # input, hidden layers, output layer
        
        # define weight and persistent activations
        self.weights = np.array([np.zeros([self.nn_shape[layer]+1, self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
        
        self.recurrent_weights = np.array([np.zeros([self.nn_shape[layer], self.nn_shape[layer]]) for layer, _ in enumerate(self.nn_shape[1:])])
        
        self.time_const = np.array([np.zeros([self.nn_shape[layer]]) for layer, _ in enumerate(self.nn_shape[1:])])
        
        self.prev_activation = np.array([np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)])
        
        # initialise weights including the bias
        for layer_nr, _ in enumerate(self.nn_shape[:-1]):
            for neuron in xrange(self.nn_shape[layer_nr]+1):
                for weight in xrange(self.nn_shape[layer_nr+1]):
                    # uniform distribution [-1,1]
                    self.weights[layer_nr][neuron][weight] = 2*random.random() - 1.
        
        # initialise recurrent weights
        for layer_nr, _ in len(self.nn_shape[1:]):
            for neuron in xrange(self.nn_shape[layer_nr]):
                for weight in xrange(self.nn_shape[layer_nr]):
                    # uniform distribution [-1,1]
                    self.recurrent_weights[layer_nr][neuron][weight] = 2*random.random() - 1.
        
        # initialise time_const
        for layer_nr, _ in len(self.nn_shape[1:]):
            for neuron in xrange(self.nn_shape[layer_nr]):
                # uniform distribution [0,1]
                self.weights[layer_nr][neuron][weight] = random.random()

    def predict(self, inputs):
    # TODO COMPLETE
        # activations include bias
        activation = np.array([np.zeros(self.nn_shape[layer]+1) for layer, _ in enumerate(self.nn_shape)])
        # assign inputs
        activation[0][:-1] = np.array(inputs[:self.nn_shape[0]])
        for layer_nr, weights_layer in enumerate(self.weights):
            activation[layer_nr][-1] = 1 # set bias activation
            activation[layer_nr+1][:-1] = activation[layer_nr].dot(weights_layer[:-1])

            # add recurrent path
            for neuron, w in enumerate(weights_layer[-1][:-1]):
                activation[layer_nr+1][neuron] += w*self.prev_activation[layer_nr+1][neuron]

            # apply relu activation function to hidden layers
            if layer_nr < len(self.weights)-1:
                np.maximum(activation[layer_nr+1], 0., activation[layer_nr+1])

        self.prev_activation = np.copy(activation)
        
        #print 'activation', activation
        
        return activation[-1][:-1]  # ignore bias neuron in output layer
        
    def load_weights(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            
            # first row contains the size of the nn
            self.nn_shape = [int(float(i)) for i in next(reader)]  # input, hidden layers, output layer
            
            # define weight and persistent activations
            self.weights = np.array([np.zeros([self.nn_shape[layer]+2, self.nn_shape[layer+1]]) for layer, _ in enumerate(self.nn_shape[:-1])])
            
            self.prev_activation = np.array([np.zeros(self.nn_shape[layer]+1) for layer, _ in enumerate(self.nn_shape)])
            
            layer = 0
            neuron = 0
            weight = 0
            for row in reader:
                for w in row:
                    if weight and weight % len(self.weights[layer][neuron]) is 0:
                        neuron += 1
                        weight = 0
                    if neuron and neuron % len(self.weights[layer]+1) is 0:
                        layer += 1
                        neuron = 0
                    self.weights[layer][neuron][weight] = w
                    weight += 1

    def mutSet(individual):
        """Mutation that pops or add an element."""
        for i in range(len(individual[0].weights)):
            for j in range(len(individual[0].weights[i])):
                if random.random() < MUTATION_RATE:
                    # mutate value in range [-w - 0.1, 2w + 0.1]
                    individual[0].weights[i][j] = (3*random.random() - 1)*individual[0].weights[i][j] + (2*random.random() - 1)*0.2
        return individual,
