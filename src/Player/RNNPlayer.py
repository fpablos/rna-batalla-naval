import gym
from gym import spaces
import numpy as np

class RNNPlayer:

    def __init__(self, env):
        self.env = env

    def rnn(self, state, out_path):
        #TODO: Read the model and use it to eval the next hit
        return self.env

    def run(self):
        return self.env.step()

    # Use the RNN to predict a move against a player and make that move
    def move(self):
        return self.env.step(self.rnn(self.env.attack_board, ''))