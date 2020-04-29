# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import time

import numpy as np
from collections import defaultdict, deque
from quoridor import Quoridor
from policy_value_net import PolicyValueNet

from mcts import MCTSPlayer
from pure_mcts import MCTSPlayer as MCTSPure
from minimax import MinimaxPlayer


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from constant import *

iter_count = 0

writer = SummaryWriter()

class TrainPipeline(object):
    def __init__(self, init_model=None):
        self.game = Quoridor()


        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 5
        self.kl_targ = 0.02
        self.check_freq = 10
        self.game_batch_num = 2000
        self.best_win_ratio = 0.0
        self.start_time = str(time.strftime('%m-%d-%h-%H-%M', time.localtime(time.time())))


        self.old_probs = 0
        self.new_probs = 0

        self.first_trained = False

        if init_model:
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet()

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=N_PLAYOUT, is_selfplay=True, test_condition=False)

    def get_wide_equi_data(self, play_data):

        self.orig_state_hist = deque(maxlen=HISTORY_LEN * 2)
        self.h_state_hist = deque(maxlen=HISTORY_LEN * 2)
        self.v_state_hist = deque(maxlen=HISTORY_LEN * 2)
        self.hv_state_hist = deque(maxlen=HISTORY_LEN * 2)

        extend_data = []

        for i, (state, mcts_prob, winner) in enumerate(play_data):

            game_info = state[1]

            state = state[0][-2:,:,:]

            wall_state = state[0,:,:]
            pos_state = state[1,:,:]

            # horizontally flipped game

            flipped_wall_state = np.fliplr(wall_state)
            flipped_pos_state = np.fliplr(pos_state)


            h_equi_state = np.stack([flipped_wall_state, flipped_pos_state])


            h_equi_mcts_prob = np.copy(mcts_prob)

            h_equi_mcts_prob[11] = mcts_prob[10]  # SE to SW
            h_equi_mcts_prob[10] = mcts_prob[11]  # SW to SE
            h_equi_mcts_prob[9] = mcts_prob[8]    # NE to NW
            h_equi_mcts_prob[8] = mcts_prob[9]    # NW to NE
            h_equi_mcts_prob[7] = mcts_prob[6]    # EE to WW
            h_equi_mcts_prob[6] = mcts_prob[7]    # WW to EE
            h_equi_mcts_prob[3] = mcts_prob[2]    # E to W
            h_equi_mcts_prob[2] = mcts_prob[3]    # W to E

            h_wall_actions = h_equi_mcts_prob[12:12 + (BOARD_SIZE-1) ** 2].reshape(BOARD_SIZE-1, BOARD_SIZE-1)
            v_wall_actions = h_equi_mcts_prob[12 + (BOARD_SIZE-1) ** 2:].reshape(BOARD_SIZE-1, BOARD_SIZE -1)

            flipped_h_wall_actions = np.fliplr(h_wall_actions)
            flipped_v_wall_actions = np.fliplr(v_wall_actions)

            h_equi_mcts_prob[12:] = np.hstack([flipped_h_wall_actions.flatten(), flipped_v_wall_actions.flatten()])

            # Vertically flipped game


            v_flipped_wall_state = np.flipud(wall_state)
            v_flipped_pos_state = -1 * np.flipud(pos_state)

            v_equi_state = np.stack([v_flipped_wall_state, v_flipped_pos_state])



            v_equi_mcts_prob = np.copy(mcts_prob)

            v_equi_mcts_prob[11] = mcts_prob[9]  # SE to NE
            v_equi_mcts_prob[10] = mcts_prob[8]  # SW to NW
            v_equi_mcts_prob[9] = mcts_prob[11]  # NE to SE
            v_equi_mcts_prob[8] = mcts_prob[10]  # NW to SW
            v_equi_mcts_prob[5] = mcts_prob[4]   # NN to SS
            v_equi_mcts_prob[4] = mcts_prob[5]   # SS to NN
            v_equi_mcts_prob[1] = mcts_prob[0]   # N to S
            v_equi_mcts_prob[0] = mcts_prob[1]   # S to N

            h_wall_actions = v_equi_mcts_prob[12:12 + (BOARD_SIZE-1) ** 2].reshape(BOARD_SIZE-1, BOARD_SIZE-1)
            v_wall_actions = v_equi_mcts_prob[12 + (BOARD_SIZE-1) ** 2:].reshape(BOARD_SIZE-1, BOARD_SIZE -1)

            flipped_h_wall_actions = np.flipud(h_wall_actions)
            flipped_v_wall_actions = np.flipud(v_wall_actions)

            v_equi_mcts_prob[12:] = np.hstack([flipped_h_wall_actions.flatten(), flipped_v_wall_actions.flatten()])

            ## Horizontally-vertically flipped game


            hv_flipped_wall_state = np.flipud(np.fliplr(wall_state))
            hv_flipped_pos_state = -1 * np.flipud(np.fliplr(pos_state))

            hv_equi_state = np.stack([hv_flipped_wall_state, hv_flipped_pos_state])

            hv_equi_mcts_prob = np.copy(mcts_prob)

            hv_equi_mcts_prob[11] = mcts_prob[8]  # SE to NW
            hv_equi_mcts_prob[10] = mcts_prob[9]  # SW to NE
            hv_equi_mcts_prob[9] = mcts_prob[10]  # NE to SW
            hv_equi_mcts_prob[8] = mcts_prob[11]  # NW to SE
            hv_equi_mcts_prob[7] = mcts_prob[6]   # EE to WW
            hv_equi_mcts_prob[6] = mcts_prob[7]   # WW to EE
            hv_equi_mcts_prob[5] = mcts_prob[4]   # NN to SS
            hv_equi_mcts_prob[4] = mcts_prob[5]   # SS to NN
            hv_equi_mcts_prob[3] = mcts_prob[2]   # E to W
            hv_equi_mcts_prob[2] = mcts_prob[3]   # W to E
            hv_equi_mcts_prob[1] = mcts_prob[0]   # N to S
            hv_equi_mcts_prob[0] = mcts_prob[1]   # S to N

            h_wall_actions = hv_equi_mcts_prob[12:12 + (BOARD_SIZE-1) ** 2].reshape(BOARD_SIZE-1, BOARD_SIZE-1)
            v_wall_actions = hv_equi_mcts_prob[12 + (BOARD_SIZE-1) ** 2:].reshape(BOARD_SIZE-1, BOARD_SIZE -1)

            flipped_h_wall_actions = np.fliplr(np.flipud(h_wall_actions))
            flipped_v_wall_actions = np.fliplr(np.flipud(v_wall_actions))

            hv_equi_mcts_prob[12:] = np.hstack([flipped_h_wall_actions.flatten(), flipped_v_wall_actions.flatten()])



            if len(self.orig_state_hist) == 0:
                for j in range(HISTORY_LEN):
                    self.orig_state_hist.extend(state)
                    self.h_state_hist.extend(h_equi_state)
                    self.v_state_hist.extend(v_equi_state)
                    self.hv_state_hist.extend(hv_equi_state)
            else:
                self.orig_state_hist.extend(state)
                self.h_state_hist.extend(h_equi_state)
                self.v_state_hist.extend(v_equi_state)
                self.hv_state_hist.extend(hv_equi_state)


            state = np.vstack([list(self.orig_state_hist)])
            h_equi_state = np.vstack([list(self.h_state_hist)])
            v_equi_state = np.vstack([list(self.v_state_hist)])
            hv_equi_state = np.vstack([list(self.hv_state_hist)])


            flipped_game_info = np.copy(game_info)
            flipped_game_info[1] = game_info[0]
            flipped_game_info[0] = game_info[1]
            flipped_game_info[2] = game_info[3]
            flipped_game_info[3] = game_info[2]

            extend_data.append((state, game_info, mcts_prob, winner))
            extend_data.append((h_equi_state, game_info,  h_equi_mcts_prob, winner))
            extend_data.append((v_equi_state, flipped_game_info, v_equi_mcts_prob, winner * -1))
            extend_data.append((hv_equi_state, flipped_game_info, hv_equi_mcts_prob, winner * -1))


        return extend_data


    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            play_data = self.get_wide_equi_data(play_data)


            self.data_buffer.extend(play_data)
            print("{}th game finished. Current episode length: {}, Length of data buffer: {}".format(i, self.episode_len, len(self.data_buffer)))

    def policy_update(self):

        #dataloader = DataLoader(self.data_buffer, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

        mini_batch = random.sample(self.data_buffer, BATCH_SIZE)
        state = [data[0] for data in mini_batch]
        game_info = [data[1] for data in mini_batch]
        mcts_prob = [data[2] for data in mini_batch]
        winner = [data[3] for data in mini_batch]


        old_probs, old_v = self.policy_value_net.policy_value(state, game_info)

        for i in range(NUM_EPOCHS):
            valloss, polloss, entropy = self.policy_value_net.train_step(state, game_info, mcts_prob, winner, self.learn_rate * self.lr_multiplier)
            self.new_probs, new_v = self.policy_value_net.policy_value(state, game_info)

            global iter_count

            iter_count += 1

            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(self.new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break


        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5


        writer.add_scalar("Val Loss/train", valloss.item(), iter_count)
        writer.add_scalar("Policy Loss/train", polloss.item(), iter_count)
        writer.add_scalar("Entropy/train", entropy.item(), iter_count)
        writer.add_scalar("LR Multiplier", self.lr_multiplier, iter_count)
        writer.add_scalar("Total Loss/train", (valloss.item() + polloss.item()), iter_count)


        return valloss.item(), polloss.item(), entropy.item()



    def policy_evaluate(self, n_games, player):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=N_PLAYOUT, is_selfplay=False)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_test_play(current_mcts_player, player, is_shown=0, first= i % 2)
            win_cnt[winner] += 1
            print("{}th evaluation game finished and won {} games out of {} games".format(i, win_cnt[1], n_games))

        win_ratio = 1.0 * (win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie: {}".format(N_MCTS_PLAYOUT, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


    def run(self):
        try:

            # win_ratio = self.policy_evaluate(n_games=1)
            pure_mcts_player = MCTSPure(c_puct=self.c_puct, n_playout=N_MCTS_PLAYOUT)

            minimax_player = MinimaxPlayer(depth=2)


            #win_ratio = self.policy_evaluate(1, pure_mcts_player)
            #writer.add_scalar("Win Ratio against pure MCTS", win_ratio, 0)
            #win_ratio = self.policy_evaluate(1, minimax_player)
            #writer.add_scalar("Win Ratio against miniamx player", win_ratio, 0)


            self.collect_selfplay_data(20)
            count = 0
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)    # collect_s
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > BATCH_SIZE:
                    valloss, polloss, entropy = self.policy_update()
                    print("VALUE LOSS: %0.3f " % valloss, "POLICY LOSS: %0.3f " % polloss, "ENTROPY: %0.3f" % entropy)

                if (i + 1) % self.check_freq == 0:
                    count += 1
                    print("current self-play batch: {}".format(i + 1))
                    # win_ratio = self.policy_evaluate()
                    # Add generation to filename
                    win_ratio = self.policy_evaluate(5, pure_mcts_player)
                    writer.add_scalar("Win Ratio against pure MCTS", win_ratio, i)
                    win_ratio = self.policy_evaluate(5, minimax_player)
                    writer.add_scalar("Win Ratio against miniamx player", win_ratio, i)

                    self.policy_value_net.save_model('model_' + str(count) + '_' + str("%0.3f_" % (valloss+polloss)) + "_BOARD_SIZE_" + str(BOARD_SIZE) + "_start_time_" + self.start_time )
        except KeyboardInterrupt:
            print('\n\rquit')


# Start
if __name__ == '__main__':

    training_pipeline = TrainPipeline(init_model=None)
    training_pipeline.run()
