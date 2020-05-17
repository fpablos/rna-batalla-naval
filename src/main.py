import argparse
import numpy as np
from .Game import Game, letter_to_coords
from .Player import Player

winner = {(True, False): 'Computer',
          (False, True): 'Player'}

# Get the input move from the player
def execute_player_move(player_env):
    p_move = np.zeros(shape=[player_env.size, player_env.size])
    x, y = player_input(player_env)
    p_move[x, y] = 1
    return p_move


# Validate the input move
def player_input(player_env):
    success = False
    while not success:
        ltr, nbr = input("Enter letter: ").upper(), input("Enter number: ")
        try:
            x, y = letter_to_coords(ltr, nbr)
            while player_env.attack_board.get_board()[x, y] != 0:
                x, y = player_input()
            success = True
        except:
            print("Invalid Input!")
            continue
    return x, y

# Takes size paramater (the board will be size * size) and ships as an array of lengths)
def start(board_size, ships):

    # Build the AI boards and init the AI
    env = Game(board_size, ships)
    computer = Player.factory('monte_carlo', env, 10000)

    # Build the Player boards
    player = Game(board_size, ships)

    # Cycle through each players turn until a player wins
    c_done = False
    p_done = False
    while not c_done and not p_done:
        c_state, c_outcome, c_done = computer.move()
        p_state, p_outcome, p_done = player.step(execute_player_move(player))

        c_state.print_board(f"=Your Board (Computer Target Ships)= [Last Outcome: {c_outcome}]")
        p_state.print_board(f"=Your Target Ships= [Last Outcome: {p_outcome}]")

    print("="*10 + "GAME OVER" + "="*10)
    print(f"The winner is: {winner[(c_done, p_done)]}")


if __name__ == '__training__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', help='The size of the board, default: 10', default=10)
    parser.add_argument('--ship_sizes', help='Array of ship sizes to randomly place, default: "5,4,3,2,2"', default='5,4,3,2,2')

    args = parser.parse_args()

    try:
        start(args.board_size, [int(x) for x in args.ship_sizes.split(',')])
    except:
        print("Invalid Arguments! =(")
        exit(1)
