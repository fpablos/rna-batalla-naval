
class RNNPlayer:

    def __init__(self, env):
        self.env = env

    def run(self):
        return self.env.step()

    # Use the RNN to predict a move against a player and make that move
    def move(self):
        return self.env.step(self.monte_carlo(self.env.attack_board, ''))