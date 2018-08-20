
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import ntpath

from model import quad_landing
from nn import nn, rnn, ctrnn, load_nn

from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_population(population, offspring=None):
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
        agent[0].save_weights('{}{}_weights.csv'.format(log_dir, i), overwrite=True)

def test_agents(weights_fp):
    agents = [join(weights_fp, f) for f in listdir(weights_fp) if isfile(join(weights_fp, f))]
    
    test_alt = 8.
    vel = []
    tim = []
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
        
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
        
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
        
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
        
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
        
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT), wind=wind)

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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT))
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
            observation, _, done, _ = env.step(actor.predict(observation, env.DT), wind=wind)

            #env.render()
            obs.append(observation.copy())
            D.append(env.state[:])
            y.append(env.y.copy())
            t.append(env.t)
            w.append(wind)
            #observation[0] = 0.
            #observation[1] = 0.
            #observation = [0., 0.]
        print max_wind
        passed = True
        if y[-1][0] > 0.3 or y[-1][1] < -0.3 or y[-1][1] > 0.:
            passed = False

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time [s]')
        ax.set_ylim([0., 10.])
        ax.plot(t, obs)
        ax.plot(t, D)
        ax.plot(t, w)

        # invert velocity sign so it fits in plot
        y = np.array(y)
        y[:,1] = -y[:,1]
        ax.plot(t, y)
        plt.title(ntpath.basename(agent_fp) + ' ' + str(passed))
        
        plt.legend(['obs D', 'obs D dot', 'ground truth D', 'ground truth D dot', 'wind', 'altitude', 'velocity', 'acceleration'])

        print('touched down at {} m/s in {} s'.format(env.y[1], env.t))
        
        if 0:
          D = np.array(D)
          ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          #D = np.array(obs)
          #ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          
          #print np.correlate(y[:,2], D[:,0])/len(y[:,2]), np.correlate(y[:,2], D[:,1])/len(y[:,2])
          
    plt.show()
    
def test_agent(agent_fp, env):
    actor = load_nn(agent_fp)
    actor.reset()

    done = False
    observation = env.reset()
    env.set_h0(8.)

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
        observation, _, done, _ = env.step(actor.predict(observation, env.DT), wind=wind)
        env.render()

        obs.append(observation.copy())
        D.append(env.state[:])
        y.append(env.y.copy())
        t.append(env.t)
        w.append(wind)
        #observation[0] = 0.
        #observation[1] = 0.
        #observation = [0., 0.]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('time [s]')
    ax.set_ylim([0., 10.])
    ax.plot(t, obs)
    ax.plot(t, D)
    ax.plot(t, w)

    # invert velocity sign so it fits in plot
    y = np.array(y)
    y[:,1] = -y[:,1]
    ax.plot(t, y)
    plt.title(ntpath.basename(agent_fp))
    
    plt.legend(['obs D', 'obs D dot', 'ground truth D', 'ground truth D dot', 'wind', 'altitude', 'velocity', 'acceleration'])

    print('touched down at {} m/s in {} s'.format(env.y[1], env.t))
    
    if 0:
      D = np.array(D)
      ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
      #D = np.array(obs)
      #ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
      
      #print np.correlate(y[:,2], D[:,0])/len(y[:,2]), np.correlate(y[:,2], D[:,1])/len(y[:,2])
    
    plt.show()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hello.')
    parser.add_argument("-a", "--agent", type=str, default='', help="The full path to the desired agent")
    args = parser.parse_args()

    test_agent(args.agent, quad_landing(delay=3, noise=0.1, visualize=True))

