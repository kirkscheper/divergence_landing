from past.builtins import xrange

import types
import tempfile
import time
from collections import deque
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/ext/gym")
import gym
from gym import spaces
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class quad_landing(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }
    # delay is time delay in time steps
    # noise_b is the std of divergence sensor noise
    # noise_p is the std of divergence sensor noise proportional to the divergence
    # thrust_tc is the motor thrust spin up time constant
    def __init__(self, delay=3, noise=0.1, noise_p=0.1, thrust_tc=0.02, dt=0.02, computational_delay_prob=0., visualize=False):
        assert delay > 0
        assert noise >= 0.
        assert thrust_tc >= 0.
        assert dt >= 0.

        self.G = 9.81
        self.MAX_H = 15.
        self.MIN_H = 0.05
        self.MAX_T = 30.

        self.DT = dt        
        self.thrust_tc = thrust_tc
        self.visualize = visualize
        self.delay = delay
        self.computational_delay_prob = computational_delay_prob
        self.noise_b_sigma = noise    # applied to divergence
        self.noise_p_sigma = noise_p    # applied to divergence

        self.viewer = None

        self.state = [0., 0.]   # divergence, divergence derivative
        self.obs = deque(maxlen=self.delay)

        obs = self.reset()

        self.action_space = spaces.Box(low=-0.8*self.G, high=0.5*self.G, shape=(1,), dtype='float32')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype='float32')

        if self.visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('time [s]')
            ax.set_ylabel('D')
            ax.set_xlim([0,self.MAX_T])
            ax.set_ylim([-0.5,3.])
            
            self.line_D_delayed = Line2D([], [], color='black')
            self.line_D = Line2D([], [], color='red')
            ax.add_line(self.line_D)
            ax.add_line(self.line_D_delayed)

    def set_h0(self, h):
        self.y[0] = h
        self.MAX_H = h + 5.

    def get_observation(self):
        # compute current ground truth state parameters
        D = -self.y[1]/(self.y[0] if abs(self.y[0]) > 1e-5 else 1e-5)
        D_dot = (D - self.state[0]) / self.DT
        self.state[:] = [D, D_dot]

        # perform observation of state
        D += np.random.normal(0., self.noise_b_sigma) + D*np.random.normal(0., self.noise_p_sigma)
        D_dot = (D - self.obs[-1][0]) / self.DT
        # low pass filter D dot
        #D_dot = self.obs[-1][1] + self.DT*((D - self.obs[-1][0]) / self.DT - self.obs[-1][1]) / (self.DT + 0.1)

        self.obs.append([D, D_dot])

        # print self.obs[0], self.state

        return np.array(self.obs[0]) # return delayed observation

    # delay thrust and ass spin up function
    def dy_dt(self, thrust_cmd):
        # action is delta thrust from hover in m/s2
        #self.thrust_cmd = np.clip(thrust_cmd, self.action_space.low, self.action_space.high)[0]
        # disable control for 1 second for RNN to settle
        if self.t < 1.:
            self.thrust_cmd = 0.
        else:
            if thrust_cmd > self.action_space.high:
                self.thrust_cmd = self.action_space.high
            elif thrust_cmd < self.action_space.low:
                self.thrust_cmd = self.action_space.low
            else:
                self.thrust_cmd = thrust_cmd

        # apply spin up to rotors
        #print thrust_cmd, self.y[2], (1./0.05)*(thrust_cmd - self.y[2])
        return np.array([self.y[1], self.y[2], (self.thrust_cmd - self.y[2])/(self.DT + self.thrust_tc)])

    def step(self, action, wind=0., computational_delay_prob=None):
        if computational_delay_prob is None:
            computational_delay_prob = self.computational_delay_prob
        # if dt large run two integration steps to still have stable rotor spin up dynamics
        #if self.DT > 0.02:
        #    self.y += (self.dy_dt(action) + [0., wind, 0.])*self.DT/2.
        #    self.y += (self.dy_dt(action) + [0., wind, 0.])*self.DT/2.
        #else:
        self.y += (self.dy_dt(action) + [0., wind, 0.])*self.DT
        self.t += self.DT

        if self.y[0] < self.MIN_H or self.t >= self.MAX_T or self.y[0] > self.MAX_H:
            if self.y[0] < self.MIN_H:
                self.y[0] = self.MIN_H
            elif self.y[0] > self.MAX_H:
                self.y[0] = self.MAX_H

            reward = 1/(self.t*self.y[1]*self.y[1]+0.01) - self.y[0]
            done = True
        else:
            reward = -1.
            done = False

        #print reward,self.t,self.y[1]
        
        # implement random delays in processing
        if(np.random.random_sample() < self.computational_delay_prob):
            self.get_observation()
            return self.step(action, wind=wind, computational_delay_prob=0.)

        return self.get_observation(), reward, done, {}

    def reset(self):
        if self.visualize and self.viewer is not None:
            self.line_D_delayed([],[])
            self.line_D([],[])

        self.t = 0.
        # state is [height, velocity, effective thrust]
        self.y = np.array([5., 0., 0.])
        # reset deque
        self.obs.clear()
        self.obs.append([0.,0.])
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
            screen_height = 900

            world_width = 10.
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

        self.line_D_delayed.set_data(np.append(self.line_D_delayed.get_xdata(), self.t), np.append(self.line_D_delayed.get_ydata(), self.obs[0][0]))
        self.line_D.set_data(np.append(self.line_D.get_xdata(), self.t), np.append(self.line_D.get_ydata(), self.obs[-1][0]))
        plt.pause(0.001)
        #self.line.append([self.y[1], self.y[0]])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    def seed(self, seed=None): return []

class quad_landing_aug(quad_landing):
    def __init__(self):
        return

