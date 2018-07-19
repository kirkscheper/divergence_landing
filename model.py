from past.builtins import xrange

import types
import tempfile
import time
from collections import deque
import numpy as np
import csv

import gym
from gym import spaces
import random

import os
from os import listdir
from os.path import isfile, join
                    
class quad_landing(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }
    # delay is time delay in time steps
    # noise is the std of divergence sensor noise
    # thrust_tc is the motor thrust spin up time constant
    def __init__(self, delay=1, noise=0.1, thrust_tc=0.025, dt=0.025, visualize=False):
        assert delay > 0
        assert noise >= 0.

        self.G = 9.81
        self.MAX_H = 15.
        self.MIN_H = 0.05
        self.MAX_T = 30.

        self.DT = dt        
        self.thrust_tc = thrust_tc
        self.visualize = visualize
        self.delay = delay
        self.noise_sigma = noise    # applied to divergence

        self.viewer = None

        self.state = [0., 0.]   # divergence, divergence derivative
        self.obs = deque(maxlen=self.delay)

        obs = self.reset()

        self.action_space = spaces.Box(low=-0.9*self.G, high=1.5*self.G, shape=(1,), dtype='float32')
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
        D_dot = (D - self.state[0]) / self.DT
        self.state = [D, D_dot]

        # perform observation of state
        D += np.random.normal(0., self.noise_sigma)
        if len(self.obs) > 0:
          D_dot = (D - self.obs[-1][0]) / self.DT
          # low pass filter D dot
          #D_dot = self.obs[-1][1] + self.DT*((D - self.obs[-1][0]) / self.DT - self.obs[-1][1]) / (self.DT + 0.1)
        else:
          D_dot = 0.
          
        self.obs.append([D, D_dot])
        
        # print self.obs[0], self.state

        return np.append(np.array(self.obs[0]), self.DT) # return delayed observation

    # delay thrust and ass spin up function
    def dy_dt(self, thrust_cmd):
        # action is delta thrust from hover in m/s2
        thrust_cmd = np.clip(thrust_cmd, self.action_space.low, self.action_space.high)
        # disable control for 1 second for RNN to settle
        if self.t < 1.:
            thrust_cmd = 0.
        
        # apply spin up to rotors
        #print thrust_cmd, self.y[2], (1./0.05)*(thrust_cmd - self.y[2])
        return np.array([self.y[1], self.y[2], (thrust_cmd - self.y[2])/(self.DT + self.thrust_tc)])

    def step(self, action, wind=0.):
        # if dt large run two integration steps to still have stable rotor spin up dynamics
        if self.DT > 0.02:
            self.y += (self.dy_dt(action) + [0., wind, 0.])*self.DT/2.
            self.y += (self.dy_dt(action) + [0., wind, 0.])*self.DT/2.
        else:
            self.y += (self.dy_dt(action) + [0., wind, 0.])*self.DT
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
        # state is [height, velocity, effective thrust]
        self.y = np.array([5., 0., 0.])
        self.obs.clear()
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

