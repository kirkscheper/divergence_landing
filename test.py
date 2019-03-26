
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import ntpath

from model import quad_landing
from nn import nn, rnn, ctrnn, load_nn

from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt

ax = None
def plot_population(population, offspring=None, pareto_front_size=0, lim=[[0,0],[0,0],[0,0]]):
    global ax    # Needed to modify global copy of globvar
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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
    
    if max(lim[0]) > 0:
        ax.set_xlim(lim[0])
    if max(lim[1]) > 0:
        ax.set_ylim(lim[1])
    if max(lim[2]) > 0:
        ax.set_zlim(lim[2])

    ax.set_xlabel('time')
    ax.set_ylabel('h')
    ax.set_zlabel('v')

    plt.pause(0.001)

    log_dir = 'logs/temp/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for i, agent in enumerate(population):
        agent[0].save_weights('{}{}_weights.csv'.format(log_dir, i), overwrite=True)

def plot_selection(population, pareto_front_size=0, lim=[[0,0],[0,0],[0,0]]):
    global ax    # Needed to modify global copy of globvar
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if pareto_front_size == 0:
        pareto_front_size = len(population)

    fitness_pop=[]
    for ind in population:
        fitness_pop.append(ind.fitness.values)
    fitness_pop = np.asarray(fitness_pop)
    ax.scatter(fitness_pop[:,0], fitness_pop[:,1], fitness_pop[:,2], color="k")

    if max(lim[0]) > 0:
        ax.set_xlim(lim[0])
    if max(lim[1]) > 0:
        ax.set_ylim(lim[1])
    if max(lim[2]) > 0:
        ax.set_zlim(lim[2])

    for txt, f in enumerate(fitness_pop[:pareto_front_size]):
        x, y, _ = proj3d.proj_transform(f[0], f[1], f[2], ax.get_proj())
        ax.annotate(txt, (x, y))

    ax.set_xlabel('time')
    ax.set_ylabel('h')
    ax.set_zlabel('v')

    plt.pause(0.001)

from itertools import chain
def log_population_performance(population, folder_location):
    env = quad_landing(delay=3, noise=0.1)
    for i, ind in enumerate(population):
        ind[0].reset()
        wind = 0

        done = False
        observation = env.reset()
        env.set_h0(4.)

        if not os.path.exists('{}/performance'.format(folder_location)):
            os.makedirs('{}/performance'.format(folder_location))
        with open('{}/performance/{}.txt'.format(folder_location, i), 'w') as fp:
            while not done:
                wind += (np.random.normal(0., 0.1)-wind)*env.DT/(env.DT+0.1)
                observation, _, done, _ = env.step(ind[0].predict(observation, env.t), wind=wind)

                np.savetxt(fp, [list(chain(*[[env.t], env.y[:], env.state[:], observation[:], [wind]]))], fmt="%s")


def log_population(population, folder_location):
    plt.cla()
    fitness_pop=[]
    for ind in population:
        fitness_pop.append(ind.fitness.values)
    fitness_pop = np.asarray(fitness_pop)

    with open('{}/pop_fitness.txt'.format(folder_location), 'w') as fp:
        np.savetxt(fp, fitness_pop, fmt="%s")

    log_population_performance(population, folder_location)


def test_agents(weights_fp):
    agents = [join(weights_fp, f) for f in listdir(weights_fp) if join(weights_fp, f).endswith(".csv")]
    agents.sort()
    test_alt = 8.
    vel = []
    tim = []
    # test 4m performance
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)

    # check that all file
    for agent_fp in agents:
        filename, file_extension = os.path.splitext(agent_fp)
        if file_extension != '.csv':
            agents.remove(agent_fp)
        
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)

    test_alt = 4.
    # test 4m performance
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test no noise
    vel.append([])
    tim.append([])
    env = quad_landing(delay=1, noise=0.)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test max noise/delay
    vel.append([])
    tim.append([])
    env = quad_landing(delay=4, noise=0.2)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test different dt
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1, dt=0.04)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
        # test external perturbations
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        wind = 0
        while not done:
            wind += (np.random.normal(0., 0.2)-wind)*env.DT/(env.DT+0.1)
            observation, _, done, _ = env.step(actor.predict(observation, env.t), wind=wind)

        vel[-1].append(env.y[1])
        tim[-1].append(env.t)

    # test no D
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
            observation[0] = 0
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test no Ddot
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor = load_nn(agent_fp)
        actor.reset()
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation, env.t))
            observation[1] = 0
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)

    ticks = [[0,1,2,3,4,5,6],['8m','4m','no noise','max noise','dt','wind','no D','no Ddot']]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('velocity [m/s]')
    plt.xticks(ticks[0], ticks[1])
    ax.plot(vel)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('time [s]')
    plt.xticks(ticks[0], ticks[1])
    ax.plot(tim)
    
    vel = np.array(vel)
    tim = np.array(tim)

    # percentage difference for noise and time
    vel_performance = np.max(abs(vel[0:5]),axis=0)
    tim_performance = 100.*(sum(tim[1:5])/3. - tim[1]) / tim[1]
    #print vel_performance #tim_performance

    #plt.show()
    #quit()

    for agent, agent_fp in enumerate(agents):
        #if tim_performance[agent] > 10.:
        #    continue

        if vel_performance[agent] > 0.3:
            continue

        #print( 'testing', agent_fp)
        actor = load_nn(agent_fp)
        actor.reset()

        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        obs = []
        D = []
        y = []
        a = []
        t = []
        w = []
        wind = 0
        max_wind = 0
        while not done:
            wind += (np.random.normal(0., 0.1)-wind)*env.DT/(env.DT+0.1)
            max_wind = max([wind, max_wind], key=abs)
            observation, _, done, _ = env.step(actor.predict(observation, env.t), wind=wind)

            #env.render()
            obs.append(observation.copy())
            D.append(env.state[:])
            y.append(env.y.copy())
            t.append(env.t)
            w.append(wind)
            #observation[0] = 0.
            #observation[1] = 0.
            #observation = [0., 0.]
        #print max_wind
        passed = True
        if y[-1][0] > 0.3 or y[-1][1] < -0.3 or y[-1][1] > 0.:
            passed = False
            
        if t[-1] > 10:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time [s]')
        ax.set_ylim([-.5, 4.])
        obs = np.array(obs)
        ax.plot(t, obs[:,0])
        ax.plot(t, D)
        ax.plot(t, w)

        # invert velocity sign so it fits in plot
        y = np.array(y)
        y[:,1] = -y[:,1]
        ax.plot(t, y)
        plt.title(ntpath.basename(agent_fp) + ' ' + str(passed))
        
        #plt.legend(['obs D', 'obs D dot', 'ground truth D', 'ground truth D dot', 'wind', 'altitude', 'velocity', 'acceleration'])
        plt.legend(['obs D', 'ground truth D', 'ground truth D dot', 'wind', 'altitude', 'velocity', 'acceleration'])

        print('touched down at {} m/s in {} s'.format(env.y[1], env.t))
        
        if 0:
          D = np.array(D)
          ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          #D = np.array(obs)
          #ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          
          #print np.correlate(y[:,2], D[:,0])/len(y[:,2]), np.correlate(y[:,2], D[:,1])/len(y[:,2])
          
    plt.show()

import csv
def test_agent_sensitivity(weights_fp):
    agents = [join(weights_fp, f) for f in listdir(weights_fp) if isfile(join(weights_fp, f))]
    
    # Get all nn files only
    for agent_fp in agents:
        filename, file_extension = os.path.splitext(agent_fp)
        if file_extension != '.csv':
            agents.remove(agent_fp)

    # Get non-dominated front
    #reader = csv.reader(open("{}/pop_fitness.txt".format(weights_fp)), delimiter=" ")
    #i = 0
    #fitnesses = []
    #for row in reader:
    #    fitnesses.append(np.asarray(row, dtype=np.float64, order='C'))
    #fitnesses = np.asarray(fitnesses)

    #for i, fitness in enumerate(fitnesses):
    #    b1 = np.where(fitnesses[:,0] < fitness[0])[0]
    #    b2 = np.where(fitnesses[:,2] < fitness[2])[0]
    #    if np.size([x for x in b1 if x in b2]) >0:
    #       fitnesses = fitnesses[0:i,:]
    #        break

    #num_non_dominated = np.shape(fitnesses)[0] 

    agents.sort()
    zero_length = len(agents[0])
    agents.sort(key=lambda x: x[-13] if np.size(x) == zero_length else x[-14:-12])
    #agents = agents[0:num_non_dominated]

    vel = []
    tim = []
    h = []

    # test 4m performance
    test_alt = 4.
    num_runs = 50
    env = quad_landing(delay=3)
    env.set_h0(test_alt)

    for agent_fp in agents:
        vel.append([])
        tim.append([])
        h.append([])

        actor = load_nn(agent_fp)

        for i in xrange(num_runs):
            np.random.seed(i)

            env = quad_landing(delay=int(num_runs/10) + 2)
            env.set_h0(test_alt)

            actor.reset()
            done = False
            observation = env.reset()

            while not done:
                observation, _, done, _ = env.step(actor.predict(observation, env.t))

            vel[-1].append(env.y[1])
            tim[-1].append(env.t)
            h[-1].append(env.y[0])

    with open('{}/sensitivity_v.txt'.format(weights_fp), 'w') as fp:
        np.savetxt(fp, vel, fmt="%s")
        
    with open('{}/sensitivity_t.txt'.format(weights_fp), 'w') as fp:
        np.savetxt(fp, tim, fmt="%s")
        
    with open('{}/sensitivity_h.txt'.format(weights_fp), 'w') as fp:
        np.savetxt(fp, h, fmt="%s")

def test_agent(agent_fp, env, visualize=True):
    actor = load_nn(agent_fp)
    actor.reset()

    done = False
    observation = env.reset()
    env.set_h0(4.)

    obs = []
    D = []
    y = []
    a = []
    t = []
    w = []
    wind = 0
    max_wind = 0
    T_hist = [[],[],[],[],[]]
    
    basepath = os.path.dirname(os.path.abspath(__file__))
    with open('{}/logs/trajectory.txt'.format(basepath), 'w') as fp:
        while not done:
            wind += (np.random.normal(0., 0.1)-wind)*env.DT/(env.DT+0.1)
            max_wind = max([wind, max_wind], key=abs)
            T = actor.predict(observation, env.t)
            T_hist[0].append(observation[0])
            T_hist[1].append(observation[1])
            T_hist[2].append(env.state[0])
            T_hist[3].append(env.state[1])
            T_hist[4].append(T[0])

            observation, _, done, _ = env.step(T, wind=wind)

            np.savetxt(fp, [list(chain(*[[env.t], env.y[:], env.state[:], observation[:], [wind]]))], fmt="%s")

            #if visualize:
            #    env.render()

            obs.append(observation.copy())
            D.append(env.state[:])
            y.append(env.y.copy())
            t.append(env.t)
            w.append(wind)
            #observation[0] = 0.
            #observation[1] = 0.
            #observation = [0., 0.]

    map_nn(args.agent, T_hist, visualize=visualize)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time [s]')
        ax.set_ylim([-.5, 4.])
        obs = np.array(obs)
        ax.plot(t, obs[:,0])
        ax.plot(t, D)
        ax.plot(t, w)
    
        # invert velocity sign so it fits in plot
        y = np.array(y)
        y[:,1] = -y[:,1]
        ax.plot(t, y)
        plt.title(ntpath.basename(agent_fp))
    
        #plt.legend(['obs D', 'obs D dot', 'ground truth D', 'ground truth D dot', 'wind', 'altitude', 'velocity', 'acceleration'])
        plt.legend(['obs D', 'ground truth D', 'ground truth D dot', 'wind', 'altitude', 'velocity', 'acceleration'])

        print('touched down at {} m/s in {} s'.format(env.y[1], env.t))

        if 0:
          D = np.array(D)
          ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          #D = np.array(obs)
          #ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          
          #print np.correlate(y[:,2], D[:,0])/len(y[:,2]), np.correlate(y[:,2], D[:,1])/len(y[:,2])
 
    plt.show()

def map_nn(agent_fp, T_hist=None, visualize=True):
    actor = load_nn(agent_fp)
    wait = False
    if actor.__class__.__name__ != 'nn':
        wait = True

    actor.reset()

    grid = np.arange(-10,10.2,0.2)
    T = np.zeros([np.size(grid), np.size(grid)])

    i = 0
    j = 0
    for D in grid:
        for Ddot in grid:
            observation = [D, Ddot]
            if wait:
                for x in xrange(200):
                    actor.predict(observation, 0.)
            T[j,i] = actor.predict(observation, 0.)
            j += 1
        j = 0
        i += 1

    basepath = os.path.dirname(os.path.abspath(__file__))
    with open('{}/logs/thrust_response.txt'.format(basepath), 'w') as fp:
        np.savetxt(fp, T, fmt="%s")

    with open('{}/logs/thrust_history.txt'.format(basepath), 'w') as fp:
        np.savetxt(fp, T_hist, fmt="%s")

    min_ind = [(np.abs(T)).argmin() / np.size(grid), (np.abs(T)).argmin() % np.size(grid)]
    min_ind = np.clip(min_ind, 5, np.size(grid)-5)
    range0 = range(min_ind[0] - 5, min_ind[0] + 5)
    range1 = range(min_ind[1] - 5, min_ind[1] + 5)

    zero_thrust = T[np.size(grid)/2,np.size(grid)/2]
    Dgain = np.mean(np.gradient(T[range0,min_ind[1]])) / (grid[1] - grid[0])
    setpoint = -zero_thrust / Dgain

    print 'T0: {}, D gradient: {}, D set point: {}, {}'.format(zero_thrust, Dgain, setpoint, grid[min_ind])
    print 'Ddot gradient: {}'.format(np.mean(np.gradient(T[min_ind[0],range1])) / (grid[1] - grid[0]))

    X, Y = np.meshgrid(grid, grid)

    if visualize:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, T)
        ax.set_zlim([-10,10])

        if T_hist is not None:
            ax.plot(T_hist[2], T_hist[3], T_hist[4], 'k')

        plt.title(ntpath.basename(agent_fp))
        plt.xlabel('D')
        plt.ylabel('Ddot')

        fig = plt.figure()
        plt.pcolormesh(X,Y,T, cmap=plt.cm.jet, vmin=-8, vmax=5)
        plt.colorbar()

        plt.title(ntpath.basename(agent_fp))
        plt.xlabel('D')
        plt.ylabel('Ddot')
        #plt.zlabel('T')

        if T_hist is not None:
            plt.plot(T_hist[0], T_hist[1], 'm')
            plt.plot(T_hist[2], T_hist[3], 'k')

        plt.show()

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_output(agent_fp, env):
    actor = load_nn(agent_fp)
    actor.reset()

    done = False
    observation = env.reset()
    observation = [1.,1.]
    env.set_h0(4.)
    
    for i in xrange(20):
        print actor.predict(observation, env.t)

    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hello.')
    parser.add_argument("-t", "--test", type=str, default='agent', help="Test to run (agent, agents, print)")
    parser.add_argument("-a", "--agent", type=str, default='', help="The full path to the desired agent")
    parser.add_argument("-v", "--visualize", type=str2bool, nargs='?', const=True, default='true', help="Whether or not the script should visualize the results")
    args = parser.parse_args()

    if args.test == "agent":
        test_agent(args.agent, quad_landing(visualize=args.visualize), visualize=args.visualize)
    elif args.test == "agents":
        test_agent_sensitivity(args.agent)
    elif args.test == "print":
        print_output(args.agent, quad_landing(visualize=False))

