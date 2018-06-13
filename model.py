
import types
import tempfile
import time
from collections import deque
import numpy as np

import keras.models
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate

import gym
from gym import spaces

import matplotlib.pyplot as plt

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def get_actor():
    #make_keras_picklable()
    input = Input(shape=(2,), name='input')

    x = Dense(8, activation='relu', bias_initializer='RandomNormal')(input)
    x = Dense(1, activation='linear')(x)
    actor = Model(inputs=input, outputs=x)
    print(actor.summary())

    # we don't optimize the model with Keras but to set to feedfoward mode
    # we have to compile the model
    actor.compile(optimizer='sgd', loss='mse')

    return actor

def get_critic(actor):
    action_input = Input(shape=(2,), name='action_input')
    input = Input(shape=(1,), name='input')

    x = Concatenate()([action_input, input])
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, input], outputs=x)

    return critic, action_input

class quad_landing(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }
    def __init__(self, delay=1, noise=0.1, visualize=False):
        assert delay > 0
        assert noise >= 0.

        self.G = 9.81
        self.MAX_H = 15.
        self.MIN_H = 0.05
        self.DT = 0.025
        self.MAX_T = 30.

        self.visualize = visualize
        self.delay=delay

        self.noise_sigma = noise    # applied to divergence

        self.viewer = None

        self.state = [0., 0.]   # divergence, divergence derivative
        self.obs = deque(maxlen=self.delay)

        obs = self.reset()

        self.action_space = spaces.Box(low=0., high=3*self.G, shape=(1,), dtype='float32')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype='float32')
        
        if self.visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('time [s]')
            ax.set_ylabel('D')
            ax.set_xlim([0,self.MAX_T])
            ax.set_ylim([-0.1,2.])
            self.plot_data, = ax.plot([],[])

    def set_h0(self, h):
        self.y[0] = h
        self.MAX_H = h + 5.

    def get_observation(self):
        # compute current ground truth state parameters
        D = -self.y[1]/(self.y[0] if np.abs(self.y[0]) > 1e-5 else 1e-5)
        D_dot = D - self.state[0]
        self.state = [D, D_dot]

        # perform observation of state
        D += np.random.normal(0., self.noise_sigma)
        D_dot = D - (self.obs[-1][0] if len(self.obs) > 0 else D)
        self.obs.append([D, D_dot])

        return np.array(self.obs[0]) # return delayed observation

    def dy_dt(self, thrust):
        # action is delta thrust from hover in m/s2
        action = thrust + self.G
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return np.array([self.y[1], action - self.G])

    def step(self, action):
        self.y += self.dy_dt(action)*self.DT
        self.t += self.DT

        if self.y[0] < self.MIN_H or self.y[0] > self.MAX_H or self.t >= self.MAX_T:
            done = True
        else:
            done = False

        if self.y[0] < 0.:
            self.y[0] = 0.

        if done:
            reward = 1/(self.t*self.y[1]*self.y[1]+0.01) - self.y[0]
            #print reward,self.t,self.y[1]
        else:
            reward = -1.

        return self.get_observation(), reward, done, {} 

    def reset(self):
        if self.visualize and self.viewer is not None:
            self.plot_data.set_xdata([])
            self.plot_data.set_ydata([])

        self.t = 0.
        # state is [height, velocity]
        self.y = np.array([5., 0.])
        return self.get_observation()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            screen_width = 200
            screen_height = 800

            world_width = 6.
            self.render_scale = screen_height/world_width

            self.viewer = rendering.Viewer(screen_width, screen_height)

            quadSize = 0.3*self.render_scale
            # add Delfly
            l,r,t,b = -quadSize/2, quadSize/2, quadSize/2, -quadSize/2
            quad = rendering.FilledPolygon([(l,b), (0,t), (0,t), (r,b)])  # triangle
            self.quadTrans = rendering.Transform()
            quad.add_attr(self.quadTrans)
            self.viewer.add_geom(quad)

            self.divergence = []

        self.quadTrans.set_translation(100, self.y[0]*self.render_scale)

        self.plot_data.set_xdata(np.append(self.plot_data.get_xdata(), self.t))
        self.plot_data.set_ydata(np.append(self.plot_data.get_ydata(), self.obs[0]))
        plt.pause(0.001)
        #self.line.append([self.y[1], self.y[0]])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    def seed(self, seed=None): return []

class quad_landing_aug(quad_landing):
    def __init__(self):
        return

