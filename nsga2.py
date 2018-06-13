#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json

import numpy as np
from copy import copy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from keras.models import save_model, load_model, clone_model

from model import quad_landing, get_actor, get_critic

MUTATION_RATE = 1.

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", array.array, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 30

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def nn_copy(nn_individual):
    model_copy = clone_model(nn_individual[0])
    model_copy.set_weights(nn_individual[0].get_weights())

    nn_copy = creator.Individual()
    nn_copy.append(model_copy)
    nn_copy.fitness = copy(nn_individual.fitness)

    return nn_copy

def evalLanding(individual):
    dyn = quad_landing(delay=3, noise=0.2)
    
    h0 = [2., 5., 10.]

    t_score = 0
    h_score = 0
    v_score = 0

    for i in range(len(h0)):
        dyn.reset()
        dyn.set_h0(h0[i])
        done = False
        while not done:
            _, _, done, _ = dyn.step(-1.)#individual[0].predict(np.array([dyn.obs[0]]))[0][0])

        t = dyn.t
        h = dyn.y[0]
        v = dyn.y[1]

        # penalize not landing, here only the hieght matters
        if t >= dyn.MAX_T or h >= dyn.MAX_H:
            v = 10.
            t = dyn.MAX_T
        
        # don't differentiate hieght score between sucessful individuals
        if h <= dyn.MIN_H:
            h = 0.
            # don't differentiate velocity score between sucessful individuals
            if np.abs(v) <= 0.05:
                v = 0.

        # penalize high speed crashing, not a viable solution
        if v < -2.:
            h = dyn.MAX_H
            t = dyn.MAX_T

        t_score += t
        h_score += h
        v_score += np.abs(v)

    # minimize time to end of sim, final height and velocity
    return t_score, h_score, v_score

def cxSet(ind1, ind2):
    return ind1, ind2
    
def mutSet(individual):
    """Mutation that pops or add an element."""
    # trainable_count = int(np.sum([K.count_params(p) for p in set(individual[0].trainable_weights)]))
    weights = individual[0].get_weights()
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.random() < MUTATION_RATE:
                # mutate value in range [-w - 0.01, 2w + 0.01]
                weights[i][j] = 3*weights[i][j]*random.random() - weights[i][j] + (random.random()*0.02 - 0.01)
    individual[0].set_weights(weights)
    return individual,

toolbox.unregister("clone")
toolbox.register("clone", nn_copy)

toolbox.register("actor", get_actor)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.actor, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalLanding)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)

    NGEN = 25
    MU = 100
    CXPB = 0.
    MUTPB = 0.

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0, 11.0]))

    return pop, logbook
        
if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    
    pop, stats = main()
    # pop.sort(key=lambda x: x.fitness.values)
    
    # print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    
    # import matplotlib.pyplot as plt
    # import numpy
    
    # front = numpy.array([ind.fitness.values for ind in pop])
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.axis("tight")
    # plt.show()
