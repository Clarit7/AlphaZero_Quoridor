import numpy as np
import copy
import random


from constant import *
from functools import reduce

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    """

    def __init__(self, parent, prior_p, state, action, depth=0):
        self._parent = parent
        self._children = {}  #
        self._n_visits = 0
        self._state = state
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._action = action
        self._depth = 0
        self._discount_factor = 0.997
        # self._game = game

    def expand(self, action_priors, is_selfplay):
        duplicated_node = False
        parent_node = None
        parent_state = None

        #zip_length = 0
        action_priors = list(action_priors)
        noise_prob = np.random.dirichlet(0.3 * np.ones(len(action_priors)))

        for i, (action, prob) in enumerate(action_priors):

            """
            if action < 12:

                # Code for restrict dummy expand

                duplicated_node = False

                # copy game - step action - get state after step(action) end
                c_game = copy.deepcopy(game)
                c_game.step(action)
                next_state = c_game.state()

                # if 'self' is not root node
                if self._parent is not None:
                    parent_node = self._parent  # get parent node
                    parent_state = parent_node._state  # get parent node state

                # Compare all states in nodes and next state
                while parent_node is not None:
                    if np.array_equal(parent_state, next_state):
                        duplicated_node = True
                        break
                    else:
                        # get parent-parent node and parent-parent node state
                        parent_node = parent_node._parent
                        if parent_node is not None:
                            parent_state = parent_node._state

            """

            if is_selfplay and self.is_root():
                prob = 0.75 * prob + 0.25 * noise_prob[i]


            if action not in self._children:
                self._children[action] = TreeNode(self, prob, None, action, self.depth()+1)

    def select(self, c_puct):

        max_value = max([act_node[1].get_value(c_puct) for act_node in self._children.items()])
        max_acts = [act_node for act_node in self._children.items() if act_node[1].get_value(c_puct) == max_value ]

        return random.choice(max_acts)
        #return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        """
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, reward):


        if self._parent:
            self._parent.update_recursive(-reward  * self._discount_factor)

        self.update(reward)

    def get_value(self, c_puct):
        """
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

        return self._Q + self._u

    def is_leaf(self):
        """
        """
        return self._children == {}

    def depth(self):
        return self._depth

    def is_root(self):
        return self._parent is None

    def get_parent(self):
        return self._parent


class MCTS(object):
    """
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1800, is_selfplay=True):
        """
        """
        self._root = TreeNode(None, 1.0, None, None)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._is_selfplay = is_selfplay

    # Fix : get current_player param info when the first simulation started.
    def _playout(self, game):
        """
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break

            action, node = node.select(self._c_puct)
            game.step(action)  #
        # state = game.state()

        action_probs, leaf_value = self._policy(game)
        end, winner = game.has_a_winner()

        if not end:
            # Add an incompleted code to make pawn avoid dead-end section.
            """
            if np.sum(game.actions()[:4]) <= 1:
                leaf_value = -1.0 if game.get_current_player == current_player else 1.0
            else:
            """
            node.expand(action_probs, self._is_selfplay)
        else:
            leaf_value = 1.0 if winner == game.get_current_player() else -1.0  # Fix bug that all winners are current player

        node.update_recursive(-leaf_value)

    def get_move_probs(self, game, temp=1, time_step=0):
        """
        """
        for n in range(self._n_playout):
            game_copy = copy.deepcopy(game)
            # state = game.state()
            # state_copy = copy.deepcopy(state)
            self._playout(game_copy)

        act_visits = [(act, node._n_visits, node._Q) for act, node in self._root._children.items()]
        acts, visits, q_values = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10 ))

        """
        visits = np.array(visits)

        if time_step < TAU_THRES:
            act_probs = visits / visits.sum()
        else:
            act_probs = np.zeros(len(visits))
            max_idx = np.argwhere(visits == visits.max())

            action_index = max_idx[np.random.choice(len(max_idx))]
            act_probs[action_index] = 1
        """

        # q_vals = [node._Q for act, node in self._root._children.items()]
        # print("-" * 30)
        # print("q_vals : ", q_vals)
        # print("-" * 30)
        return acts, act_probs, q_values

    def update_with_move(self, last_move, state):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, state, last_move)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    #
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=1, test_condition=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, is_selfplay)
        self._is_selfplay = is_selfplay
        self._test_condition = test_condition
        self._scenario = 0
        self._move70p = 0
        self._activate_dirichlet = 0
        self._random_choose = 0

    #
    def set_player_ind(self, p):
        self.player = p

    #
    def reset_player(self):
        self.mcts.update_with_move(-1, None)

    # Choose an action during the play
    def choose_action(self, game, temp=1e-3, return_prob=0, time_step=0):
        sensible_moves = game.actions()
        move_probs = np.zeros(12 + (BOARD_SIZE - 1) ** 2 * 2)
        q_vals = np.zeros(12 + (BOARD_SIZE - 1) ** 2 * 2)


        if len(sensible_moves) > 0:
            acts, probs, q_values = self.mcts.get_move_probs(game, temp, time_step)
            move_probs[list(acts)] = probs
            q_vals[list(acts)] = q_values
            state = game.state()

            if self._is_selfplay:
                if self._test_condition:
                    move = self.test_action_choose(acts, probs, time_step)
                else:
                    probs = 0.8 * probs + 0.2 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                    # move = acts[np.argmax(probs)]
                    move = np.random.choice(acts, p=probs)

                self.mcts.update_with_move(move, state)
            else:
                print(probs)
                move = acts[np.argmax(probs)]
                # move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1, state)

            if return_prob:
                return move, move_probs, q_vals
            else:
                return move
        else:
            print("WARNING: the board is full")

    def test_action_choose(self, acts, probs, time_step):

        # choose force move in early game to prevent overfitting
        if time_step == 1:
            self._scenario = np.random.randint(6)
            self._move70p = np.random.randint(10)
            self._activate_dirichlet = np.random.randint(10)
            self._random_choose = np.random.randint(10)

            print("================== Current Game Random Setting Info ===================")
            if self._scenario < 2:
                print("Each player move forward in first step (33.3%)")
            elif self._scenario == 2:
                print("Player 1 move forward in first step scenario (16.7%)")
            elif self._scenario == 3:
                print("Player 2 move forward in first step (16.7%)")
            else:
                print("Free move scenario (33.3%)")

            if self._move70p < 3:
                print("70% prob to choose pawn move (30%)")
            else:
                print("Normal move (70%)")

            if self._activate_dirichlet < 3:
                print("Deactivate dirichlet noise (30%)")
            else:
                print("Activate dirichlet noise (70%)")

            if self._random_choose < 3:
                print("Choose argmax move based on the move probability (30%)")
            else:
                print("Choose random move based on the move probability (70%)")

            print("=======================================================================")

        if self._activate_dirichlet < 3:
            probs = 0.8 * probs + 0.2 * np.random.dirichlet(0.3 * np.ones(len(probs)))

        if self._scenario < 2 and time_step < 3:  # each player move forward in first step scenario
            if time_step == 1:
                move = 0
            else:
                move = 1
        elif self._scenario == 2 and time_step == 1:  # p1 move forward in first step scenario
            move = 0
        elif self._scenario == 3 and time_step == 2 and 1 in acts:  # p2 move forward in first step scenario
            move = 1
        else:
            if self._move70p < 3:  # choose pawn move in 70% prob case
                wall_remain = False
                for i, v in enumerate(acts):
                    if v > 11:
                        if np.random.randint(10) < 7:
                            pawn_acts = acts[:i]
                            pawn_probs = softmax(probs[:i])
                            if self._random_choose:  # random choose case
                                move = np.random.choice(pawn_acts, p=softmax(pawn_probs))
                            else:  # max prob choose case
                                move = pawn_acts[np.argmax(pawn_probs)]
                        else:
                            wall_acts = acts[i:]
                            wall_probs = softmax(probs[i:])
                            if self._random_choose < 3:  # random choose case
                                move = np.random.choice(wall_acts, p=softmax(wall_probs))
                            else:  # max prob choose case
                                move = wall_acts[np.argmax(wall_probs)]
                        wall_remain = True
                        break

                if not wall_remain:  # if no wall remains, choose action normally(= choose pawn move)
                    if self._random_choose < 3:  # random choose case
                        move = np.random.choice(acts, p=probs)
                    else:  # max prob choose case
                        move = acts[np.argmax(probs)]
            else:  # choose action normally
                if self._random_choose < 3:  # random choose case
                    move = np.random.choice(acts, p=probs)
                else:  # max prob choose case
                    move = acts[np.argmax(probs)]

        return move

    def __str__(self):
        return "MCTS {}".format(self.player)
