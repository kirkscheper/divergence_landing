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

from builtins import input

import array
import random
import json

import numpy as np
import time
import argparse
import os

from math import sqrt, ceil

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from scoop import futures

from model import quad_landing, nn, rnn, plot_population, test_agents

MUTATION_RATE = 0.7

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

def evalLanding(individual):
    # set noise paramters
    # time delay range [1,4] time steps
    # divegence sensor noise [0,0.15]s-1
    # thrust time constant [0, 0.1]s
    # time step [1/30, 1/50]s
    dyn = quad_landing(delay=ceil(4*np.random.random_sample()),
      noise=0.15*np.random.random_sample(),
      thrust_tc=0.15*np.random.random_sample(),
      dt=1/ceil(20*np.random.random_sample() + 30))
    
    h0 = [2., 5., 10.]

    t_score = 0.
    h_score = 0.
    v_score = 0.

    for i in range(len(h0)):
        obs = dyn.reset()
        dyn.set_h0(h0[i])
        done = False
        while not done:
            obs, _, done, _ = dyn.step(individual[0].predict(obs))

        t = dyn.t
        h = dyn.y[0]
        v = dyn.y[1]

        # penalize not landing, here only the hieght matters
        if t >= dyn.MAX_T or h >= dyn.MAX_H:
            v = -10.
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

neural_type = ctrnn

toolbox.register("individual", tools.initRepeat, creator.Individual, neural_type, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalLanding)
toolbox.register("mate", cxSet)
toolbox.register("mutate", neural_type.mutSet)
toolbox.register("select", tools.selNSGA2)

toolbox.register("map", futures.map)

def main(seed=None):
    random.seed(seed)

    NGEN = 500
    MU = 100
    
    log_interval = 25

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

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
    
    hof.update(pop)
    
    log_dir = 'logs/{}/'.format(time.strftime('%y%m%d-%H%M%S'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Begin the generational process
    for gen in range(1, NGEN+1):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Revaluate the individuals in last population
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evaluate the new offspring
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        plot_population(pop, offspring)
        
        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring)+len(pop), **record)
        print(logbook.stream)
        
        if gen % log_interval == 0 or gen == NGEN:
            os.makedirs('{}{}'.format(log_dir, gen))
            for i, agent in enumerate(pop):
                agent[0].save_weights('{}{}/{}_weights.csv'.format(log_dir, gen, i), overwrite=True)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0, 11.0]))
    
    os.makedirs('{}hof'.format(log_dir))
    for i, agent in enumerate(hof):
        agent[0].save_weights('{}hof/{}_weights.csv'.format(log_dir, i), overwrite=True)
    
    return pop, logbook
        
if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['evolve','train', 'test'], default='evolve')
    #parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--visualize', action='store_const', const=1, default=0)
    args = parser.parse_args()
    
    if args.mode == 'evolve':
        pop, stats = main()
        
    elif args.mode == 'test':
        if args.weights is None:
            raise ValueError('You must provide a model to train with the weights input')

        test_agents(args.weights)

    input('Press any key to exit')
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
