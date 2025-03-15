from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inference:

    def inference(self,
                  state: np.ndarray,
                  current_player: int,
                  use_gpu: bool = False) -> Union[np.ndarray, float]:
        """
        Runs the model to get policy and value for a single state.
        """
        # Flatten the board to shape (9,)
        board_input = (current_player * state).astype(np.float32)
        board_input = torch.from_numpy(board_input).unsqueeze(0)  # shape (1,9)

        if use_gpu:
            board_input = board_input.cuda()

        # Evaluate
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(board_input)
            policy = self.softmax(policy_logits)
            # policy_logits -> shape (1,9)
            # value -> shape (1,1)

        # Convert to numpy
        policy = policy[0].cpu().numpy()  # shape (9,)
        value = value[0, 0].cpu().numpy().item()  # scalar

        return policy, value


class TicTacToeNet(nn.Module, Inference):

    def __init__(self, state_size, output_size):
        super().__init__()
        # Input dimension = 3*3 = 9 (board cells)
        # We'll embed each cell’s value (–1, 0, +1) directly.
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # Policy head (outputs a logit for each action 0..8)
        self.policy_head = nn.Linear(64, output_size)

        # Value head (outputs a single scalar in –1..+1, so we often do a tanh)
        self.value_head = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        :param x: a float tensor of shape [batch_size, 9], representing board states
                  (flattened 3x3). x can be –1, 0, or +1 in each cell.
        :return: (policy_logits, value)
          policy_logits shape: [batch_size, 9]
          value shape: [batch_size, 1]
        """
        # Basic MLP
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        value = torch.tanh(value)  # range in (–1, +1)

        return policy_logits, value


class OthelloNet(nn.Module, Inference):

    def __init__(self, board_size: int, action_size: int):
        """
        :param board_size: size of the board (e.g., 8 for an 8x8 board)
        :param action_size: number of actions (board_size*board_size + 1 for the "pass" move)
        """
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # Convolutional layers: input channel=1 (board values: in canonical form {-1,0,1})
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3,
            padding=1)  # output: [batch, 32, board_size, board_size]
        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3,
            padding=1)  # output: [batch, 32, board_size, board_size]

        # Compute flattened dimension dynamically.
        flatten_dim = 32 * board_size * board_size

        # Policy head: maps flattened conv output to action logits.
        self.fc_policy = nn.Linear(flatten_dim, action_size)

        # Value head: a smaller hidden layer then output a scalar value.
        self.fc_value1 = nn.Linear(flatten_dim, 32)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        :param x: tensor of shape [batch_size, board_size, board_size] with values in {-1, 0, +1}.
        :return: tuple (policy_logits, value)
                 - policy_logits: tensor of shape [batch_size, action_size]
                 - value: tensor of shape [batch_size, 1] in range (-1, +1)
        """
        # If x is missing the channel dimension, add it.
        if x.dim() == 3:
            x = x.unsqueeze(
                1)  # Now shape: [batch_size, 1, board_size, board_size]

        # Apply convolutional layers with ReLU activations.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten feature maps.
        x_flat = x.view(x.size(0), -1)

        # Policy head: outputs logits for each action.
        policy_logits = self.fc_policy(x_flat)

        # Value head: hidden layer, then a tanh squashing to produce value in (-1, +1).
        value = F.relu(self.fc_value1(x_flat))
        value = torch.tanh(self.fc_value2(value))

        return policy_logits, value
