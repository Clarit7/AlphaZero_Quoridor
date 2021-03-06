# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import time

import numpy as np
from collections import defaultdict, deque
from quoridor import Quoridor
from policy_value_net import PolicyValueNet

from mcts import MCTSPlayer


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
        self.n_playout = 200
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.kl_targ = 0.02
        self.check_freq = 10
        self.game_batch_num = 1000
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

        self.old_probs = 0
        self.new_probs = 0

        self.first_trained = False

        if init_model:
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet()

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):

        extend_data = []
        for i, (state, mcts_prob, winner) in enumerate(play_data):
            wall_state = state[:3,:BOARD_SIZE - 1,:BOARD_SIZE - 1]
            dist_state1 = np.reshape(state[(6 + (WALL_NUM + 1) * 2), :BOARD_SIZE, :BOARD_SIZE], (1, BOARD_SIZE, BOARD_SIZE))
            dist_state2 = np.reshape(state[(7 + (WALL_NUM + 1) * 2), :BOARD_SIZE, :BOARD_SIZE], (1, BOARD_SIZE, BOARD_SIZE))

            # horizontally flipped game
            flipped_wall_state = []

            for i in range(3):
                wall_padded = np.fliplr(wall_state[i])
                wall_padded = np.pad(wall_padded, (0,1), mode='constant', constant_values=0)
                flipped_wall_state.append(wall_padded)

            flipped_wall_state = np.array(flipped_wall_state)

            player_position = state[3:5, :,:]

            flipped_player_position = []
            for i in range(2):
                flipped_player_position.append(np.fliplr(player_position[i]))

            flipped_player_position = np.array(flipped_player_position)

            h_equi_state = np.vstack([flipped_wall_state, flipped_player_position, state[5:, :,:]])

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
            flipped_wall_state = []

            for i in range(3):
                wall_padded = np.flipud(wall_state[i])
                wall_padded = np.pad(wall_padded, (0,1), mode='constant', constant_values=0)
                flipped_wall_state.append(wall_padded)

            flipped_wall_state = np.array(flipped_wall_state)


            flipped_player_position = []
            for i in range(2):
                flipped_player_position.append(np.flipud(player_position[1-i]))

            flipped_player_position = np.array(flipped_player_position)

            cur_player = (np.ones((BOARD_SIZE, BOARD_SIZE)) - state[5 + 2* (WALL_NUM+1),:,:]).reshape(-1,BOARD_SIZE, BOARD_SIZE)

            v_equi_state = np.vstack([flipped_wall_state, flipped_player_position, state[5+(WALL_NUM+1):5 + 2*(WALL_NUM+1), :,:], state[5:5+(WALL_NUM+1),:,:], cur_player, dist_state2, dist_state1])
            # v_equi_state = np.vstack([flipped_wall_state, flipped_player_position, state[5:(5 + (WALL_NUM+1) * 2), :, :], cur_player, state[:(6 + (WALL_NUM + 1) * 2), :, :]])


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

            wall_state = state[:3,:BOARD_SIZE - 1,:BOARD_SIZE - 1]
            flipped_wall_state = []

            for i in range(3):
                wall_padded = np.fliplr(np.flipud(wall_state[i]))
                wall_padded = np.pad(wall_padded, (0,1), mode='constant', constant_values=0)
                flipped_wall_state.append(wall_padded)

            flipped_wall_state = np.array(flipped_wall_state)



            flipped_player_position = []
            for i in range(2):
                flipped_player_position.append(np.fliplr(np.flipud(player_position[1-i])))

            flipped_player_position = np.array(flipped_player_position)

            cur_player = (np.ones((BOARD_SIZE, BOARD_SIZE)) - state[5 + 2*(WALL_NUM+1),:,:]).reshape(-1,BOARD_SIZE, BOARD_SIZE)

            hv_equi_state = np.vstack([flipped_wall_state, flipped_player_position, state[5 + (WALL_NUM+1):5 + 2*(WALL_NUM+1), :,:], state[5:5+(WALL_NUM+1),:,:], cur_player, dist_state2, dist_state1])
            # hv_equi_state = np.vstack([flipped_wall_state, flipped_player_position, state[5:(5 + (WALL_NUM+1) * 2), :, :], cur_player, state[(6 + (WALL_NUM + 1) * 2):, :, :]])

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

            ###########

            extend_data.append((state, mcts_prob, winner))
            extend_data.append((h_equi_state, h_equi_mcts_prob, winner))
            extend_data.append((v_equi_state, v_equi_mcts_prob, winner * -1))
            extend_data.append((hv_equi_state, hv_equi_mcts_prob, winner * -1))

        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            play_data = self.get_equi_data(play_data)

            self.data_buffer.extend(play_data)
            print("{}th game finished. Current episode length: {}, Length of data buffer: {}".format(i, self.episode_len, len(self.data_buffer)))

    def policy_update(self):

        dataloader = DataLoader(self.data_buffer, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        valloss_acc = 0
        polloss_acc = 0
        entropy_acc = 0

        for i in range(NUM_EPOCHS):

            self.old_probs = self.new_probs

            if self.first_trained:
                kl = np.mean(np.sum(self.old_probs * (np.log(self.old_probs + 1e-10) - np.log(self.new_probs + 1e-10)), axis=1))
                if kl > self.kl_targ * 4:
                    break

                if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                    self.lr_multiplier /= 1.5
                elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                    self.lr_multiplier *= 1.5


            for i, (state, mcts_prob, winner) in enumerate(dataloader):
                valloss, polloss, entropy = self.policy_value_net.train_step(state, mcts_prob, winner, self.learn_rate * self.lr_multiplier)
                self.new_probs, new_v = self.policy_value_net.policy_value(state)

                global iter_count

                writer.add_scalar("Val Loss/train", valloss.item(), iter_count)
                writer.add_scalar("Policy Loss/train", polloss.item(), iter_count)
                writer.add_scalar("Entropy/train", entropy, iter_count)
                writer.add_scalar("LR Multiplier", self.lr_multiplier, iter_count)

                iter_count += 1

                valloss_acc += valloss.item()
                polloss_acc += polloss.item()
                entropy_acc += entropy.item()

            self.first_trained = True

        valloss_mean = valloss_acc / (len(dataloader) * NUM_EPOCHS)
        polloss_mean = polloss_acc / (len(dataloader) * NUM_EPOCHS)
        entropy_mean = entropy_acc / (len(dataloader) * NUM_EPOCHS)

        #explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        #explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        #print( "kl:{:.5f}, lr_multiplier:{:.3f}, value loss:{}, policy loss:[], entropy:{}".format(
        #        kl, self.lr_multiplier, valloss, polloss, entropy, explained_var_old, explained_var_new))
        return valloss_mean, polloss_mean, entropy_mean

    def run(self):
        try:
            self.collect_selfplay_data(3)
            count = 0
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)    # collect_s
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > BATCH_SIZE:
                    valloss, polloss, entropy = self.policy_update()
                    print("VALUE LOSS: %0.3f " % valloss, "POLICY LOSS: %0.3f " % polloss, "ENTROPY: %0.3f" % entropy)

                    #writer.add_scalar("Val Loss/train", valloss.item(), i)
                    #writer.add_scalar("Policy Loss/train", polloss.item(), i)
                    #writer.add_scalar("Entory/train", entropy, i)

                if (i + 1) % self.check_freq == 0:
                    count += 1
                    print("current self-play batch: {}".format(i + 1))
                    # win_ratio = self.policy_evaluate()
                    # Add generation to filename
                    self.policy_value_net.save_model('model_7x7_' + str(count) + '_' + str("%0.3f_" % (valloss+polloss) + str(time.strftime('%Y-%m-%d', time.localtime(time.time())))))
        except KeyboardInterrupt:
            print('\n\rquit')


# Start
if __name__ == '__main__':

    training_pipeline = TrainPipeline(init_model=None)
    training_pipeline.run()
