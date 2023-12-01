"""
Utils.py

Utility functions for Hanamikoji game. These are for precomputing relevant dictionaries which would take a lot of compute to do every single iteration.
Additional functions are included for create plots at the end of training.

William Dinauer, November 2023
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def get_possible_actions(state, storage, player):
        """
        Computes the possible actions that a certain player can take given the state of the game, returning them as an array.
        """
        # print(f"getting possible actions for {len(player.hand), tuple(player.moves_left)}")
        # Determine the available actions based on the current state

        # Response action
        if len(state['board']['response_buffer']) > 0:
            # Possible actions are the indices of choice
            if len(state['board']['response_buffer']) == 4:
                return [[0], [1]]
            else:
                return [[0], [1], [2]]

        # Standard action
        possible_actions = storage[len(player.hand), tuple(player.moves_left)]

        # print(f"got possible actions: {possible_actions}")

        return possible_actions

def precompute_possibilities():
    """
    Precomputes every possible move for every possible state, returning them as a dictionary
    The dictionary takes a hand size and tuple of 'moves left' as keys, which maps to the associated possible actions by index.
    """
    storage = {}
    possible_ml = []
    ml = []

    # Helper function to recursively create every possible combination of moves left (15 possibilities)
    def recursive_ml(idx):
        if idx == 5:
            if len(ml) > 0:
                possible_ml.append(ml.copy())
            return
        recursive_ml(idx+1)
        ml.append(idx)
        recursive_ml(idx+1)
        ml.remove(idx)

    # Compute every possible 'moves left' combination
    recursive_ml(1)

    # i represents hand size. The size of a hand ranges from 1 to 7 when picking a move.
    for i in range(1, 8):
        for ml in possible_ml:
            actions = []
            for move in ml:
                if move == 1:
                    for j in range(i):
                        actions.append([j])
                if move == 2:
                    for j in range(i):
                        for k in range(j+1, i):
                            actions.append((j, k))
                if move == 3:
                    for j in range(i):
                        for k in range(j+1, i):
                            for l in range(k+1, i):
                                actions.append((j, k, l))
                if move == 4:
                    for j in range(i):
                        for k in range(j+1, i):
                            for l in range(k+1, i):
                                for m in range(l+1, i):
                                    actions.append((j, k, l, m))
                                    actions.append((j, l, k, m))
                                    actions.append((j, m, k, l))
            storage[i, tuple(ml)] = actions

    return storage


def create_action_dict():
    """
    Creates and returns two dictionaries, 'action_dict' and 'int_dict'. They are the inverse of each other, used for translating a move into an integer or an integer into a move.
    There are 273 possible moves in total, so there are 273 key-value pairs in each table.

    'int_dict' takes an integer as input and outputs the associated action.
    'action_dict' takes an action as input and outputs the associated integer.
    """
    action_dict = {}
    int_dict = {}
    action = 0

    # Move 1
    for i in range(7):
        action_dict[i] = action
        int_dict[action] = [i]
        action += 1

    # Move 2
    for i in range(7):
        for j in range(i+1, 7):
            action_dict[(i, j)] = action
            int_dict[action] = (i, j)
            action += 1

    # Move 3
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                action_dict[(i, j, k)] = action
                int_dict[action] = (i, j, k)
                action += 1

    # Move 4
    for i in range(7):
        for j in range(i, 7):
            for k in range(j+1, 7):
                for m in range(k+1, 7):
                    action_dict[(i, j, k, m)] = action
                    action_dict[(i, m, j, k)] = action+1
                    action_dict[(i, k, j, m)] = action+2

                    int_dict[action] = (i, j, k, m)
                    int_dict[action+1] = (i, m, j, k)
                    int_dict[action+2] = (i, k, j, m)
                    action += 3

    return action_dict, int_dict

def plot_learning(x, scores, epsilons, filename, lines=None):
    """
    Plotting function for the scores and epsilons at the end of training
    """
    fig = plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-10000):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Average Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")
    ax2.grid(True)

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def plot_single(x, percent, filename, lines=None):
    """
    Plotting function for a single parameter
    """
    fig = plt.figure()
    
    ax = fig.add_subplot(111, label="1")
    ax.plot(x, percent, color="C0")
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Win-Percent", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    # Enable grid
    ax.grid(True)

    # Add more precision to the y-axis
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if lines is not None:
        for line in lines:
            ax.axhline(line, color='red', linestyle='--')

    plt.savefig(filename)
    plt.show()  # Optionally, display the plot