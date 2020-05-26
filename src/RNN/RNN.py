import gym
import os
from gym import spaces
import numpy as np
from stable_baselines.common.env_checker import check_env
from stable_baselines import A2C
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
from tensorflow.keras.backend import clear_session

import matplotlib.pyplot as plt
log_dir = "./data/"
best_mean_reward, n_steps, step_interval, episode_interval = -np.inf, 0, 10000, 10000

chars = {'unknown':   '■',
         'hit':       'X',
         'miss':      '□',
         'sea':       '■',
         'destroyed': '*'}


# randomly places a ship on a board
def set_ship(ship, ships, board, ship_locs):
    grid_size = board.shape[0]

    done = False
    while not done:
        init_pos_i = np.random.randint(0, grid_size)
        init_pos_j = np.random.randint(0, grid_size)

        # for a cruiser, if init_oos_i = 0, move forward horizontally (+1)
        # for a cruiser, if init_oos_j = 0, move downward vertically (+1)
        move_j = grid_size - init_pos_j - ships[ship]  # horizontal
        if move_j > 0:
            move_j = 1
        else:
            move_j = -1
        move_i = grid_size - init_pos_i - ships[ship]  # vertical
        if move_i > 0:
            move_i = 1
        else:
            move_i = -1
        # choose if placing ship horizontally or vertically
        choice_hv = np.random.choice(['h', 'v'])  # horizontal, vertical
        if choice_hv == 'h':  # horizontal
            j = [(init_pos_j + move_j * jj) for jj in range(ships[ship])]
            i = [init_pos_i for ii in range(ships[ship])]
            pos = set(zip(i, j))
            if all([board[i, j] == 0 for (i, j) in pos]):
                done = True
        elif choice_hv == 'v':
            i = [(init_pos_i + move_i * ii) for ii in range(ships[ship])]
            j = [init_pos_j for jj in range(ships[ship])]
            pos = set(zip(i, j))
            # check if empty board in this direction
            if all([board[i, j] == 0 for (i, j) in pos]):
                done = True
    # set ship - see convention
    for (i, j) in pos:
        board[i, j] = 1
        ship_locs[ship].append((i, j))

    return board, ship_locs

class BattleshipEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, enemy_board, ship_locs, grid_size, ships):
        super(BattleshipEnv, self).__init__()

        self.ships = ships
        self.size = grid_size

        # cell state encoding (empty, hit, miss)
        self.square_states = { 'unknown': 0,
                               'hit': 1,
                               'miss': -1}

        # Init boards
        self.is_enemy_set = False
        self.enemy_board = enemy_board
        self.ship_locs = ship_locs

        if self.enemy_board is None:
            self.enemy_board = 0 * np.ones((self.size, self.size), dtype='int')
            for ship in self.ships:
                self.ship_locs[ship] = []
                self.enemy_board, self.ship_locs = set_ship(ship, self.ships, self.enemy_board, self.ship_locs)
            self.is_enemy_set = True

        # reward discount
        self.rdisc = 0

        self.legal_actions = []  # legal (empty) cells available for moves
        for i in range(self.size):
            for j in range(self.size):
                self.legal_actions.append((i, j))  # this gets updated as an action is performed

        # In our case the action space is discrete: index of action
        self.action_space = spaces.Discrete(self.size * self.size)

        # The observation will be the state or configuration of the board
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.size, self.size), dtype=np.int)

    # action will be an index in action_space if from epsilon-greedy
    # or from model prediction
    def step(self, action):
        # board situation before the action
        state = self.get_board(True)
        empty_counts_pre, hit_counts_pre, miss_counts_pre = self.board_config(state)

        # action coordinates generated or predicted by the agent in the action_space
        i, j = np.unravel_index(action, (self.size, self.size))

        # lose 1 point for any action
        reward = -1

        # assign a penalty for each illegal action used instead of a legal one
        if (i, j) not in self.legal_actions:
            reward -= 2 * self.size
            action_idx = np.random.randint(0, len(self.legal_actions))

            i, j = self.legal_actions[action_idx]
            action = np.ravel_multi_index((i, j), (self.size, self.size))

        # set new state after performing action (scoring board is updated)
        self.set_state((i, j))

        # update legal actions and action_space
        self.set_legal_actions((i, j))

        # new state on scoring board - this includes last action
        next_state = self.board

        # board situation after action
        empty_counts_post, hit_counts_post, miss_counts_post = self.board_config(next_state)

        # game completed?
        done = bool(hit_counts_post == sum(self.ships.values()))

        # reward for a hit
        if hit_counts_post - hit_counts_pre == 1:
            # Update hit counts and use it to reward
            r_discount = 1  # 0.5**self.rdisc
            rp = (self.size * self.size if done else self.size)
            reward += rp * r_discount

        # setup the new reward
        reward = float(reward)

        # store the current value of the portfolio here
        info = {}

        return next_state, reward, done, info

    # Reset the state of the environment to an initial state
    def reset(self):
        self.init_board()

        return self.board

    def get_board(self, copy=False):
        if copy:
            return self.board.copy()
        else:
            return self.board

    def init_board(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)

        # Init Legal (empty) cells available for moves
        self.legal_actions = []
        for i in range(self.size):
            for j in range(self.size):
                self.legal_actions.append((i, j))  # this gets updated as an action is performed

        # generate a random board again if it was set randomly before
        if self.is_enemy_set:
            self.enemy_board = 0 * np.ones((self.size, self.size), dtype='int')
            self.ship_locs = {}
            for ship in self.ships:
                self.ship_locs[ship] = []
                self.enemy_board, self.ship_locs = set_ship(ship, self.ships, self.enemy_board, self.ship_locs)

        # Init Reward discount
        self.rdisc = 0

    def render(self, mode='human'):
        print("   ", end='')
        for i in range(self.size):
            print(i + 1, end=' ')
        print('')
        for i, x in enumerate(self.get_board()):
            print(chr(i + 65), end='  ')
            for y in x:
                print(chars[self.square_states[y]], end=' ')
            print('')


    def board_config(self, state):
        uni_states, uni_cnts = np.unique(state.ravel(), return_counts=True)
        empty_counts = uni_cnts[uni_states == self.square_states['unknown']]
        hit_counts = uni_cnts[uni_states == self.square_states['hit']]
        miss_counts = uni_cnts[uni_states == self.square_states['miss']]
        if len(empty_counts) == 0:
            empty_counts = 0
        else:
            empty_counts = empty_counts[0]

        if len(hit_counts) == 0:
            hit_counts = 0
        else:
            hit_counts = hit_counts[0]

        if len(miss_counts) == 0:
            miss_counts = 0
        else:
            miss_counts = miss_counts[0]

        return empty_counts, hit_counts, miss_counts

    # set board configuration and state value after player action
    def set_state(self, action):
        i, j = action
        if self.enemy_board[i, j] == 1:
            self.board[i, j] = self.square_states['hit']
        else:
            self.board[i, j] = self.square_states['miss']

    # set legal actions (empty board locations)
    def set_legal_actions(self, action):
        if action in self.legal_actions:
            self.legal_actions.remove(action)

def callback(_locals, _globals):
    global n_steps, best_mean_reward
    # Print stats every step_interval calls
    if (n_steps + 1) % step_interval == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            # NOTE: when done is True, timesteps are counted and reported to the log_dir
            mean_reward = np.mean(y[-episode_interval:]) # mean reward over previous episode_interval episodes
            mean_moves = np.mean(np.diff(x[-episode_interval:])) # mean moves over previous episode_interval episodes
            print(x[-1], 'timesteps') # closest to step_interval step number
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - Last mean moves per episode: {:.2f}".format(best_mean_reward,
                                                                                           mean_reward, mean_moves))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, episode_interval: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_interval = episode_interval
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model.pkl')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # NOTE: when done is True, timesteps are counted and reported to the log_dir
                mean_reward = np.mean(y[-self.episode_interval:]) # mean reward over previous episode_interval episodes
                mean_moves = np.mean(np.diff(x[-self.episode_interval:])) # mean moves over previous 100 episodes
                if self.verbose > 0:
                    print(x[-1], 'timesteps') # closest to step_interval step number
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - Last mean moves per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                   mean_reward, mean_moves))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model")
                    self.model.save(self.save_path)

        return True


"""
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
"""
def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


"""
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
"""
def plot_results(log_folder, window=100, title='Learning Curve'):

    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    y_moves = moving_average(np.diff(x), window=window)
    # Truncate x
    x = x[len(x) - len(y):]
    x_moves = x[len(x) - len(y_moves):]

    title = 'Smoothed Learning Curve of Rewards (every ' + str(window) + ' steps)'
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

    title = 'Smoothed Learning Curve of Moves (every ' + str(window) + ' steps)'
    fig = plt.figure(title)
    plt.plot(x_moves, y_moves)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Moves')
    plt.title(title)
    plt.show()

def training(board_size, ships):
    clear_session()
    dir = log_dir + str(board_size) +'x'+ str(board_size) +'/'
    # Ships
    ships={}
    # Number of moves
    num_timesteps = 1000000

    ship_locs = {}

    # Instantiate the env
    env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=board_size, ships=ships)

    # Check if the directory exists
    os.makedirs(dir, exist_ok=True)

    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # Train the agent
    model = A2C('MlpPolicy', env, verbose=0).learn(total_timesteps=num_timesteps, callback=callback)

    model.save(dir + 'battleship')

def plot_training():
    plot_results(log_dir, 10000)