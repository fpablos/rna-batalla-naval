from .Player import Player

class RNNPlayer(Player):

    def __init__(self, env, simulations):
        self.env = env
        # The number of moves to simulate
        self.move_sim = simulations

    def run(self):
        return self.env.step()

    # Use the RNN to predict a move against a player and make that move
    def move(self):
        return self.env.step(self.monte_carlo(self.env.attack_board, ''))