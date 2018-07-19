



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
    actor = rnn()

    agents = [join(weights_fp, f) for f in listdir(weights_fp) if isfile(join(weights_fp, f))]
    
    test_alt = 5.
    vel = []
    tim = []
    # test 5m performance
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)

    test_alt = 10.
    # test 10m performance
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test no noise
    vel.append([])
    tim.append([])
    env = quad_landing(delay=1, noise=0.)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test max noise/delay
    vel.append([])
    tim.append([])
    env = quad_landing(delay=4, noise=0.2)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test different dt
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1, dt=0.04)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test no D
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
            observation[0] = 0
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
        
    # test no Ddot
    vel.append([])
    tim.append([])
    env = quad_landing(delay=3, noise=0.1)
    for agent_fp in agents:
        actor.load_weights(agent_fp)
        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
            observation[1] = 0
        
        vel[-1].append(env.y[1])
        tim[-1].append(env.t)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('velocity [m/s]')
    plt.xticks([0,1,2,3,4,5],['5m','10m','no noise','max noise','dt','no D','no Ddot'])
    ax.plot(vel)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('time [s]')
    plt.xticks([0,1,2,3,4,5],['5m','10m','no noise','max noise','dt','no D','no Ddot'])
    ax.plot(tim)
    
    vel = np.array(vel)
    tim = np.array(tim)
    
    # percentage difference for noise and time
    vel_performance = max(vel[0:5])
    tim_performance = 100.*(sum(tim[1:5])/3. - tim[1]) / tim[1]
    print tim_performance

    #plt.show()
    #quit()
    
    for agent, agent_fp in enumerate(agents):
        #if tim_performance[agent] > 10.:
        #    continue
        
        if vel_performance > 0.5:
            continue

        #print( 'testing', agent_fp)
        actor.load_weights(agent_fp)

        done = False
        observation = env.reset()
        env.set_h0(test_alt)

        obs = []
        D = []
        y = []
        a = []
        t = []
        while not done:
            observation, _, done, _ = env.step(actor.predict(observation))
            #env.render()
            obs.append(observation.copy())
            D.append(env.state[:])
            y.append(env.y.copy())
            t.append(env.t)
            #observation[0] = 0.
            #observation[1] = 0.
            #observation = [0., 0.]

        if y[-1][0] > 0.3 or y[-1][1] < -0.3:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time [s]')
        ax.set_ylim([0., 10.])
        ax.plot(t, obs)
        ax.plot(t, D)
        
        # invert velocity sign so it fits in plot
        y = np.array(y)
        y[:,1] = -y[:,1]
        ax.plot(t, y)
        plt.title(agent_fp)
        
        plt.legend(['obs D', 'obs D dot', 'ground truth D', 'ground truth D dot', 'altitude', 'velocity', 'acceleration'])

        print('touched down at {} m/s in {} s'.format(env.y[1], env.t))
        
        if 0:
          D = np.array(D)
          ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          #D = np.array(obs)
          #ax.plot(t, abs(-y[:,2]/(D[:,1] - D[:,0]*D[:,0] + 0.001)))
          
          #print np.correlate(y[:,2], D[:,0])/len(y[:,2]), np.correlate(y[:,2], D[:,1])/len(y[:,2])
          
    plt.show()
