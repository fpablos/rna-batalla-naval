import argparse, traceback, sys
from Game import Game
from Player import Player
from RNN import training, plot_training

winner = {(True, False): 'Computer',
          (False, True): 'Player'}

# Takes size paramater (the board will be size * size) and ships as an array of lengths)
def start(board_size, ships, player1, player2):

    # Build the board and player1
    game1 = Game(board_size, ships)
    computer = Player.factory(player1, game1)

    # Build the board and player2
    game2 = Game(board_size, ships)
    enemy = Player.factory(player2, game2)

    # Cycle through each players turn until a player wins
    c_done = False
    e_done = False
    while not c_done and not e_done:
        c_state, c_outcome, c_done = computer.move()
        e_state, e_outcome, e_done = enemy.move()

        c_state.print_board(f"=Your Board (Computer Target Ships)= [Last Outcome: {c_outcome}]")
        e_state.print_board(f"=Your Target Ships= [Last Outcome: {e_outcome}]")

    print("="*10 + "GAME OVER" + "="*10)
    print(f"The winner is: {winner[(c_done, e_done)]}")

def rnn_training(board_size, ships):
    #print(board_size)
    #print(ships)
    training(board_size, ships)
    plot_training()

def rnn_plot_training():
    plot_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', help='The size of the board, default: 10', default=10)
    parser.add_argument('--ship_sizes', help='Array of ship sizes to randomly place, default: "5,4,3,2,2"', default='5,4,3,2,2')
    parser.add_argument('--ia_battle',  help='The final battle between monte carlo algorithm and RNA', default=False)
    parser.add_argument('--rnn',        help='Player vs RNA', default=False)
    parser.add_argument('--monte_carlo', help='Player vs RNA', default=True)
    parser.add_argument('--training', help='Training the RNA', default=False)
    parser.add_argument('--plot_training', help='Training the RNA', default=False)

    args = parser.parse_args()

    if args.plot_training:
        rnn_plot_training()
        exit()

    if args.training:
        rnn_training(int(args.board_size), [int(x) for x in args.ship_sizes.split(',')])
        exit()

    player1 = "monte_carlo"
    player2 = "human"

    if args.ia_battle:
        player1 = "monte_carlo"
        player2 = "rnn"
    elif args.rnn:
        player1 = "rnn"

    try:
        print("Chosen args: ", args.board_size, [int(x) for x in args.ship_sizes.split(',')], player1, player2)
        start(args.board_size, [int(x) for x in args.ship_sizes.split(',')], player1, player2)
    except:
        #print("Invalid Arguments! =(")
        traceback.print_exc(file=sys.stdout)
        exit(1)
