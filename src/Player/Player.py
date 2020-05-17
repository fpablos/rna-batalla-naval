# A simple static factory method.
from __future__ import generators
from .MonteCarloPlayer import MonteCarloPlayer
from .RNNPlayer import RNNPlayer

class Player(object):
    def factory(self, type, params):
        if type == 'monte_carlo':
            return MonteCarloPlayer(params)
        if type == 'rnn':
            return RNNPlayer(params)
        assert 0, "Bad Player creation " + type
    factory = staticmethod(factory)