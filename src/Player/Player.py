# A simple static factory method.
from .MonteCarloPlayer import MonteCarloPlayer
from .RNNPlayer import RNNPlayer
from .HumanPlayer import HumanPlayer

class Player(object):
    @classmethod
    def factory(cls, type, game):
        if type == 'monte_carlo':
            return MonteCarloPlayer(game, 10000)
        if type == 'rnn':
            return RNNPlayer(game)

        return HumanPlayer(game)