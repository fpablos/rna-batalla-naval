import numpy as np
from Game import letter_to_coords

class HumanPlayer:

    def __init__(self, env):
        self.env = env

    # Get the input move from the player
    def execute_player_move(self):
        p_move = np.zeros(shape=[self.env.size, self.env.size])
        x, y = self.player_input()
        p_move[x, y] = 1
        return p_move

    # Validate the input move
    def player_input(self):
        success = False
        while not success:
            ltr, nbr = input("Enter letter: ").upper(), input("Enter number: ")
            try:
                x, y = letter_to_coords(ltr, nbr)
                while self.env.attack_board.get_board()[x, y] != 0:
                    x, y = self.player_input()
                success = True
            except:
                print("Invalid Input!")
                continue
        return x, y

    def run(self):
        return True

    # Use the RNN to predict a move against a player and make that move
    def move(self):
        return self.env.step(self.execute_player_move())