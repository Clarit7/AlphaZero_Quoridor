from __future__ import print_function

from quoridor import Quoridor
from policy_value_net import PolicyValueNet

from mcts import MCTSPlayer as A_Player
from pure_mcts import MCTSPlayer as B_Player
from minimax import MinimaxPlayer as C_Player


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", "-m", help="File name")

args = parser.parse_args()


class TestTrainedAgent(object):
    def __init__(self, init_model=None, first_player=1):
        self.game = Quoridor()
        self.temp = 1.0
        self.c_puct = 5
        self.play_batch_size = 1
        self.alpha_playout = 100
        self.pure_playout = 100
        self.first = first_player

        self.alpha_player = A_Player(PolicyValueNet(model_file=init_model, use_gpu=True).policy_value_fn, c_puct=self.c_puct,
                                     n_playout=self.alpha_playout, is_selfplay=False)

        self.pure_player = A_Player(PolicyValueNet(model_file='ckpt/model_d_80_2.052__BOARD_SIZE_5_start_time_06-02-Jun-15-08.pth', use_gpu=True).policy_value_fn,
                                     c_puct=self.c_puct,
                                     n_playout=self.alpha_playout, is_selfplay=False)


        #self.alpha_player = B_Player(c_puct=1, n_playout=self.alpha_playout)

        # self.pure_player = B_Player(c_puct=5, n_playout=self.pure_playout)
        # self.pure_player = C_Player(depth=2)

        self.alpha_win_total = 0
        self.alpha_win_first = 0
        self.alpha_draw_total = 0
        self.alpha_draw_first = 0
        self.total_games = 0


    def testing_game(self, num):

        print(num, "th game started")

        game_clone = self.game.clone()

        if self.first == 3:
            if self.total_games % 2 == 0:
                winner = game_clone.start_test_play(self.minimax_player, self.pure_player,
                                                   temp=self.temp, first=1)
            else:
                winner = game_clone.start_test_play(self.minimax_player, self.pure_player,
                                                   temp=self.temp, first=2)
        else:
            winner = game_clone.start_test_play(self.minimax_player, self.pure_player,
                                                   temp=self.temp, first=self.first)

        self.total_games += 1

        if winner == 1:
            if self.total_games % 2 == 0:
                self.alpha_win_first += 1
                self.alpha_win_total += 1
            else:
                self.alpha_win_total += 1
            print("{}th/{} game finished. Winner is alpha zero player".format((self.total_games + 1), n_games))
        elif winner == 0:
            if self.total_games % 2 == 0:
                self.alpha_draw_first += 1
                self.alpha_draw_total += 1
            else:
                self.alpha_draw_total += 1
            print("{}th/{} game finished. Draw".format((i + 1), n_games))
        else:
            print("{}th/{} game finished. Winner is pure mcts player".format((self.total_games + 1), n_games))
        print("alpha zero win rate in first start : {:.2%}".format(self.alpha_win_first / (self.total_games//2 + 1)))
        print("alpha zero win rate in second start : {:.2%}".format((self.alpha_win_total - self.alpha_win_first) / ((self.total_games+1) - (self.total_games//2 +1 ) + 1e-10)))
        print("alpha zero win rate total : {:.2%}".format(self.alpha_win_total / (self.total_games + 1)))
        print("alpha zero draw rate in first start : {:.2%}".format(self.alpha_draw_first / (self.total_games//2 + 1)))
        print("alpha zero draw rate in second start : {:.2%}".format(
            (self.alpha_draw_total - self.alpha_draw_first) / ((self.total_games+1) - (self.total_games//2 + 1) + 1e-10)))
        print("alpha zero draw rate total : {:.2%}".format(self.alpha_draw_total / (self.total_games + 1)))
        print(num + " game finished.")


    def test_against_pure(self, n_games=1):
        """
        for i in range(n_games):
            if self.first == 3:
                if i % 2 == 0:
                    winner = self.game.start_test_play(self.alpha_player, self.pure_player,
                                                       temp=self.temp, first=1)
                else:
                    winner = self.game.start_test_play(self.alpha_player, self.pure_player,
                                                       temp=self.temp, first=2)
            else:
                winner = self.game.start_test_play(self.alpha_player, self.pure_player,
                                                   temp=self.temp, first=self.first)


            if winner == 1:
                if i % 2 == 0:
                    self.alpha_win_first += 1
                    self.alpha_win_total += 1
                else:
                    self.alpha_win_total += 1
                print("{}th/{} game finished. Winner is alpha zero player".format((i + 1), n_games))
            elif winner == 0:
                if i % 2 == 0:
                    self.alpha_draw_first += 1
                    self.alpha_draw_total += 1
                else:
                    self.alpha_draw_total += 1
                print("{}th/{} game finished. Draw".format((i + 1), n_games))
            else:
                print("{}th/{} game finished. Winner is pure mcts player".format((i + 1), n_games))
            print("alpha zero win rate in first start : {:.2%}".format(self.alpha_win_first / (i//2 + 1)))
            print("alpha zero win rate in second start : {:.2%}".format((self.alpha_win_total - self.alpha_win_first) / ((i+1) - (i//2 +1 ) + 1e-10)))
            print("alpha zero win rate total : {:.2%}".format(self.alpha_win_total / (i + 1)))
            print("alpha zero draw rate in first start : {:.2%}".format(self.alpha_draw_first / (i//2 + 1)))
            print("alpha zero draw rate in second start : {:.2%}".format(
                (self.alpha_draw_total - self.alpha_draw_first) / ((i+1) - (i//2 + 1) + 1e-10)))
            print("alpha zero draw rate total : {:.2%}".format(self.alpha_draw_total / (i + 1)))
        """

        winner = self.game.start_test_play(A_Player(PolicyValueNet(model_file='ckpt/model_d_5_2.767__BOARD_SIZE_5_start_time_06-02-Jun-15-08.pth', use_gpu=True).policy_value_fn,
                                     c_puct=self.c_puct, n_playout=self.alpha_playout, is_selfplay=False), self.pure_player, temp=self.temp, first=1)
        print("Winner is Minimax player")
        print("==============================================")
        winner = self.game.start_test_play(A_Player(PolicyValueNet(model_file='ckpt/model_d_95_2.127__BOARD_SIZE_5_start_time_06-02-Jun-15-08.pth', use_gpu=True).policy_value_fn,
                                     c_puct=self.c_puct, n_playout=self.alpha_playout, is_selfplay=False), self.pure_player, temp=self.temp, first=1)
        print("Winner is AlphaZero player")
        print("==============================================")

    def run(self, epoch_num):
        try:
            self.test_against_pure(n_games=epoch_num)
            # pool = multiprocessing.Pool(processes=8)
            # pool.map(self.testing_game, range(100))
            # pool.close()
            # pool.join()

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # init_model : alpha zero model file name
    # first_player : 1 - alpha zero, 2 - pure mcts, 3 - change first player when every game
    test_trained_agent = TestTrainedAgent(init_model=args.model, first_player=3)
    test_trained_agent.run(100)
