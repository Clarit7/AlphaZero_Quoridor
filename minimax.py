# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter
import random
import math


def policy_fn(game, depth):
    # return uniform probabilities and 0 score for pure MCTS

    player = game.current_player

    best = -math.inf
    best_action = game.actions()[0]

    is_over, winner = game.has_a_winner()

    if is_over or depth == 0:
        if player == 1:
            return best_action, game.dist2 - game.dist1
        else:
            return best_action, game.dist1 - game.dist2


    for action in game.actions():
        game_copy = copy.deepcopy(game)
        game_copy.step(action)
        _, value = policy_fn(game_copy, depth-1)
        value = -value
        if value > best:
            best = value
            best_action = action

    # print("Best action: ", best_action, ", Value: ", best, ", Depth: ", depth)
    # print("Action: ", game.actions())
    # print("Player: ", player, ", P1 dist: ", game.dist1, ", P2 dist: ", game.dist2)
    # print("-" * 12)

    return best_action, best


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class Minimax(object):
    def __init__(self, depth, policy_fn=policy_fn):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_fn
        self._depth = depth


    def get_move(self, game):

        best_move = self._policy(game, depth=self._depth)

        return best_move

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MinimaxPlayer(object):
    def __init__(self, depth):
        self.minimax = Minimax(depth=depth, policy_fn=policy_fn)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.minimax.update_with_move(-1)

    def choose_action(self, game):
        sensible_moves = game.actions()
        if len(sensible_moves) > 0:
            move, move_value = self.minimax.get_move(game)
            self.minimax.update_with_move(-1)
            return move, move_value
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
