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

import random
import time
import os
from os import listdir
from os.path import isfile, join
import numpy as np

import argparse

from model import quad_landing, get_actor, get_critic

from copy import copy

from keras import backend as K
from keras.utils import layer_utils as keras_utl
from keras.models import save_model, load_model, clone_model
from keras.optimizers import Adam

#from rl.memory import PrioritizedMemory
#from rl.agents import DDPGAgent
#from rl.random import GaussianWhiteNoiseProcess

if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from paralleltoolbox import ParallelToolbox

from bisect import bisect_right

#from scipy.integrate import odeint

import multiprocessing
#from scoop import futures

MUTATION_RATE = 1.

# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(64)

def nn_copy(nn_individual):
    model_copy = clone_model(nn_individual[0])
    model_copy.set_weights(nn_individual[0].get_weights())

    nn_copy = creator.Individual()
    nn_copy.append(model_copy)
    nn_copy.fitness = copy(nn_individual.fitness)

    return nn_copy

class ParetoFrontNN(tools.ParetoFront):
    def insert(self, item):
        item = nn_copy(item)
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

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
            _, _, done, _ = dyn.step(individual[0].predict(np.array([dyn.obs[0]]))[0][0])

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

from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_callback(population, offspring=None):
    plt.cla()
    fitness_pop=[]
    for ind in population:
        fitness_pop.append(ind.fitness.values)
    fitness_pop = np.asarray(fitness_pop)
    ax.scatter(fitness_pop[:,0], fitness_pop[:,1], fitness_pop[:,2])

    if offspring is not None:
        fitness=[]
        for ind in offspring:
            fitness.append(ind.fitness.values)
        fitness = np.asarray(fitness)
        ax.scatter(fitness[:,0], fitness[:,1], fitness[:,2], color="r")
    
    for txt, f in enumerate(fitness_pop):
        x, y, _ = proj3d.proj_transform(f[0], f[1], f[2], ax.get_proj())
        ax.annotate(txt, (x, y))

    ax.set_xlabel('time')
    ax.set_ylabel('h')
    ax.set_zlabel('v')

    plt.pause(0.01)

    log_dir = 'logs/temp/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for i, agent in enumerate(population):
        agent[0].save_weights('{}{}_weights.h5f'.format(log_dir, i), overwrite=True)

def evolve():
    NGEN = 25
    MU = 50
    LAMBDA = 100
    CXPB = 0.
    MUTPB = 1.

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront#ParetoFrontNN()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)#, callback=plot_callback)

    log_dir = 'logs/{}/'.format(time.strftime('%y%m%d-%H%M%S'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for i, agent in enumerate(hof):
        agent[0].save_weights('{}{}_weights.h5f'.format(log_dir, i), overwrite=True)

#def run_rl(model):
#    env = quad_landing()

#    memory = PrioritizedMemory(limit=1000)

#    sigma = np.abs((env.action_space.high - env.action_space.low)) / 2.
#    random_process = GaussianWhiteNoiseProcess(size=1, mu=0., sigma=sigma, sigma_min=sigma*0.1, n_steps_annealing=3000)

#    actor = get_actor()
#    critic, actor_input = get_critic(actor)
#    agent = DDPGAgent(nb_actions=1, actor=actor, critic=critic,
#                      critic_action_input=actor_input, memory=memory, nb_steps_warmup_critic=200,
#                      nb_steps_warmup_actor=2000, random_process=random_process, 
#                      gamma=.95, delta_clip=1., target_model_update=1e-2, batch_size=32)
#    agent.compile([Adam(lr=.0001), Adam(lr=.001)], metrics=['mape'])

#    agent.actor.load_weights(model)

#    agent.test(env, nb_episodes=1, verbose=1, visualize=True)

#    agent.fit(env, nb_steps=10000, verbose=2, log_interval=250)

#    filename, extension = os.path.splitext(model)
#    filepath = filename + '_rl' + extension
#    agent.save_weights(filepath, overwrite=True)

#    agent.test(env, nb_episodes=1, verbose=1, visualize=True)

def test_agents(weights_fp):
    env = quad_landing(delay=3)
    actor = get_actor()
    for w in actor.trainable_weights:
        print(w)

    agents = [join(weights_fp, f) for f in listdir(weights_fp) if isfile(join(weights_fp, f))]
    for agent_fp in agents:
        print( 'testing', agent_fp)
        actor.load_weights(agent_fp)
        for w in actor.get_weights():
            print(w)

        done = False
        observation = env.reset()
        env.set_h0(10.)

        obs = []
        D = []
        y = []
        t = []
        while not done:
            observation, _, done, _ = env.step(actor.predict(np.expand_dims(observation,0))[0][0])
            #env.render()
            obs.append(observation.copy())
            D.append(env.state[:])
            y.append(env.y.copy())
            t.append(env.t)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time [s]')
        ax.set_ylim([0., 5.])
        ax.plot(t, obs)
        ax.plot(t, D)
        
        # invert velocity sign so it fits in plot
        y = np.array(y)
        y[:,1] = -y[:,1]
        ax.plot(t, y)
        plt.title(agent_fp)

        v = env.y[1]        
        if env.y[0] > 0.1:
            v = 10.
        print('touched down at {} m/s in {} s'.format(v, env.t))
    plt.show()
    

def bridge_reality_gap():
    # here I want to run some supervised learning with dense rewards on the diff
    # between the simulated dynamics and the augmented simulator dynamics
    return

def evalOneMax(individual):
    return sum(individual),
    
if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['evolve','train', 'test'], default='evolve')
    #parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--visualize', action='store_const', const=1, default=0)
    args = parser.parse_args()
    
    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    #toolbox = ParallelToolbox()

    toolbox.unregister("clone")
    toolbox.register("clone", nn_copy)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, get_actor, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evalOneMax)#evalLanding)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    # Process Pool of 4 workers
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)
    
    #toolbox.register("map", futures.map)
    
    if args.mode == 'evolve':
        evolve()

    elif args.mode == 'train':
        if args.weights is None:
            raise ValueError('You must provide a model to train with the weights input')

        run_rl(args.weights)

    elif args.mode == 'test':
        if args.weights is None:
            raise ValueError('You must provide a model to train with the weights input')

        test_agents(args.weights)


