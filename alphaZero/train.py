import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from MCTS_model import MCTS
from eval import evaluate_models


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


class Trainer:

    def __init__(self, env, args, policy):
        self.env = env
        self.args = args
        self.policy = policy

        self.best_policy = copy.deepcopy(self.policy)
        self.best_policy.eval()

        self.value_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'])

        self.current_training_data = []
        self.dataloader = None

        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'current_win_rate': [],
            'best_win_rate': []
        }

        self.self_play_history = {'winner': []}

    def clean_training_data(self):
        self.current_training_data = []

    def _take_freshest_training_data(self):
        """
        Keep only the most recent max_samples training examples.
        Since we're only using fresh samples, this simply truncates
        the list to at most max_samples entries.
        """
        max_train_samples = self.args['max_train_samples']
        if len(self.current_training_data) > max_train_samples:
            self.current_training_data = self.current_training_data[
                -max_train_samples:]

    def setup_dataloader(self):
        """
        Convert self.current_training_data (list of (state, policy, value))
        into a PyTorch DataLoader for training.
        """
        self._take_freshest_training_data()

        states = []
        policies = []
        values = []
        for (s, p, v) in self.current_training_data:
            states.append(s)  # 2D array: shape (board_size, board_size)
            policies.append(p)  # policy vector, shape (action_size,)
            values.append(v)  # scalar

        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)
        values = np.array(values, dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(states),
                                torch.from_numpy(policies),
                                torch.from_numpy(values))
        self.dataloader = DataLoader(dataset,
                                     batch_size=self.args['batch_size'],
                                     shuffle=True)

    @torch.no_grad()
    def self_play(self):
        # same as rollout, but the policy is not random
        trajectory = []

        state = self.env.get_initial_state()
        player, is_terminal = 1, False
        mcts = MCTS(self.env, self.args, self.best_policy, False)
        while True:
            action_probs = mcts.policy_improve_step(
                state, player, temp=self.args['mcts_temperature'])
            trajectory.append(
                (state.copy() * player, action_probs.copy(), player))

            action = np.random.choice(self.env.action_size, p=action_probs)
            mcts.make_move(action)
            state = self.env.get_next_state(state, action, player)
            reward, is_terminal = self.env.get_value_and_terminated(
                state, action, player)

            if is_terminal:
                if reward > 0:
                    winning_player = player
                elif reward < 0:
                    winning_player = self.env.get_opponent(player)
                else:
                    winning_player = 0

                # update the rewards based on outcome
                for i, (st, pol, ply) in enumerate(trajectory):
                    outcome = 1 if ply == winning_player else (
                        -1 if winning_player != 0 else 0)
                    trajectory[i] = (st, pol, outcome)

                # Accumulate the entire game data
                self.self_play_history['winner'].append(player)
                self.current_training_data += trajectory
                return

            player = self.env.get_opponent(player)

    def train_iters(self):
        self.policy.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        dataloader_iter = infinite_dataloader(self.dataloader)

        for _ in range(self.args['train_steps_per_iter']):
            state, target_policy, target_value = next(dataloader_iter)
            # predict
            policy_logits, value = self.policy(state)
            # compute policy loss
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.mean(
                torch.sum(target_policy * log_probs, dim=-1))
            # compute value loss
            value_loss = self.value_loss(value, target_value)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        avg_total_loss = avg_policy_loss + avg_value_loss

        self.metrics_history['policy_loss'].append(avg_policy_loss)
        self.metrics_history['value_loss'].append(avg_value_loss)
        self.metrics_history['total_loss'].append(avg_total_loss)

        print(f"Policy Loss={self.metrics_history['policy_loss'][-1]:.4f}, "
              f"Value Loss={self.metrics_history['value_loss'][-1]:.4f}")

    def train(self):
        for _ in range(self.args['num_iterations']):
            # self.clean_training_data()

            # self-play
            for _ in range(self.args['num_self_play']):
                self.self_play()

            # train
            self.setup_dataloader()
            self.train_iters()

            self.eval()

    def eval(self):
        self.policy.eval()
        current_win_rate, best_win_rate = evaluate_models(
            self.env,
            self.args,
            self.policy,
            self.best_policy,
            n_matches=self.args['num_eval_matches'])

        print(
            f"Win Rate Current {current_win_rate*100:.1f}% vs Win rate Best: {best_win_rate*100:.1f}%"
        )

        self.metrics_history['current_win_rate'].append(current_win_rate)
        self.metrics_history['best_win_rate'].append(best_win_rate)

        # If better than the best model by a margin, accept new model as best
        if current_win_rate - self.args['eval_win_margin'] > best_win_rate:
            print("Replacing best model with current model")
            self.best_policy = copy.deepcopy(self.policy)
