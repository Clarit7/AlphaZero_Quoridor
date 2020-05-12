import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from constant import *

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)





class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(residual.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.size())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class policy_value_net(nn.Module):
    def __init__(self, block, inplanes, planes, stride=1):
        super(policy_value_net, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        blocks = []

        for i in range(NUM_BLOCK):
            blocks.append(block(planes, planes))

        self.layers = nn.Sequential(*blocks)

        val_dim = 4
        pol_dim = 32


        self.fc1 = nn.Linear((BOARD_SIZE * 2 - 1) ** 2 * planes, planes)
        self.fc2 = nn.Linear(planes, planes)

        self.fc3 = nn.Linear(planes + 4, planes // 4)
        self.pol_fc = nn.Linear(planes // 4, (BOARD_SIZE - 1) ** 2 * 2 + 12)
        self.val_fc = nn.Linear(planes // 4, 1)


        # policy head

        # self.policy_conv = conv1x1(planes, pol_dim)
        # self.pol_bn = nn.BatchNorm2d(pol_dim)

        # self.pol_fc = nn.Linear((BOARD_SIZE * 2 - 1) ** 2 * pol_dim + 4, (BOARD_SIZE - 1) ** 2 * 2 + 12)



        # value head

        # self.value_conv = conv1x1(planes, val_dim)
        # self.val_bn = nn.BatchNorm2d(val_dim)

        # self.val_fc = nn.Linear((BOARD_SIZE * 2 - 1) ** 2 * val_dim + 4, 64)
        # self.val_fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)


    def forward(self,x, y):


        # new_y = y.unsqueeze(2).unsqueeze(2).repeat(1, 1, BOARD_SIZE * 2 -1, BOARD_SIZE * 2 - 1)

        # x = torch.cat([x, new_y], dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layers(out)


        out = out.view(out.shape[0], -1)

        out = self.dropout(self.relu(self.fc2(self.dropout(self.relu(self.fc1(out))))))

        out = self.dropout(self.relu(self.fc3(torch.cat([out,y], dim=1))))

        # policy head


        #pol_out = self.relu(self.pol_bn(self.policy_conv(out)))
        #pol_out = self.pol_fc(torch.cat([pol_out.view(pol_out.shape[0], -1), y ], dim=1 ))


        pol_out = F.log_softmax(self.pol_fc(out), dim = 1)


        # value head

        #val_out = self.relu(self.val_bn(self.value_conv(out)))
        #val_out = self.relu(self.val_fc(torch.cat([val_out.view(val_out.shape[0], -1), y], dim = 1 )))
        #val_out = self.val_fc2(val_out)

        val_out = F.tanh(self.val_fc(out))


        return pol_out,val_out

class PolicyValueNet(object):
    def __init__(self,model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  #
        if self.use_gpu:
            self.policy_value_net = policy_value_net(BasicBlock, STATE_DIM * HISTORY_LEN,NN_DIM).cuda()
        else:
            self.policy_value_net = policy_value_net(BasicBlock, STATE_DIM * HISTORY_LEN,NN_DIM)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    def policy_value(self, state_batch, game_info):


        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            game_info = Variable(torch.FloatTensor(game_info).cuda())
            log_act_probs, value = self.policy_value_net(state_batch, game_info)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            device = torch.device("cpu")
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch, game_info)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, game):
        legal_positions = game.actions()

        current_state = np.ascontiguousarray(game.widestate2()[0]).reshape([1, (2) * HISTORY_LEN, BOARD_SIZE * 2 - 1, BOARD_SIZE * 2 - 1]) # 1 x (2 * 5) x 9 x 9

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float(), Variable(torch.from_numpy(game.additional_info())).cuda().float().view(1,-1))
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())

        if game.get_current_player() == 2:
            v_equi_mcts_prob = np.copy(act_probs)

            v_equi_mcts_prob[11] = act_probs[9]  # SE to NE
            v_equi_mcts_prob[10] = act_probs[8]  # SW to NW
            v_equi_mcts_prob[9] = act_probs[11]  # NE to SE
            v_equi_mcts_prob[8] = act_probs[10]  # NW to SW
            v_equi_mcts_prob[5] = act_probs[4]   # NN to SS
            v_equi_mcts_prob[4] = act_probs[5]   # SS to NN
            v_equi_mcts_prob[1] = act_probs[0]   # N to S
            v_equi_mcts_prob[0] = act_probs[1]   # S to N

            h_wall_actions = v_equi_mcts_prob[12:12 + (BOARD_SIZE-1) ** 2].reshape(BOARD_SIZE-1, BOARD_SIZE-1)
            v_wall_actions = v_equi_mcts_prob[12 + (BOARD_SIZE-1) ** 2:].reshape(BOARD_SIZE-1, BOARD_SIZE -1)

            flipped_h_wall_actions = np.flipud(h_wall_actions)
            flipped_v_wall_actions = np.flipud(v_wall_actions)

            v_equi_mcts_prob[12:] = np.hstack([flipped_h_wall_actions.flatten(), flipped_v_wall_actions.flatten()])

            act_probs = v_equi_mcts_prob


        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.item()
        return act_probs, value

    def train_step(self, state_batch, game_info, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            # device = torch.device("cuda:0")
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            game_info = Variable(torch.FloatTensor(game_info).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            # device = torch.device("cpu")
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch, game_info)
        value_loss = F.mse_loss(value.view(-1), winner_batch)

        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))

        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return value_loss.data, policy_loss.data, entropy.data  # Change code for newest PyTorch version

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), 'ckpt/%s.pth'%(model_file))


