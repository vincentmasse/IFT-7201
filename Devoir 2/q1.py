from poutyne import Model
from copy import deepcopy  # NEW
from scipy.stats import bernoulli

import numpy as np
import gym
import torch
import random


from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size = buffer_size
        # TODO : Add any needed attributes
        self.data = deque(maxlen=int(buffer_size))

    def store(self, element):
        """
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        """
        # TODO : Implement
        self.data.append(element)
        pass

    def get_batch(self, batch_size):
        """
        Returns a list of batch_size elements from the buffer.
        """
        # TODO : Implement
        buffer_list = list(self.data)
        return buffer_list[-batch_size:]


class DQN(Model):
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        super().__init__(*args, **kwargs)

    def get_action(self, state, epsilon):
        """
        Returns the selected action according to an epsilon-greedy policy.
        """
        # TODO: implement
        if bernoulli.rvs(epsilon):
            action = random.choice(self.actions)
        else:
            action = np.argmax(state)
        return action


    def soft_update(self, other, tau):
        """
        Code for the soft update between a target network (self) and
        a source network (other).

        The weights are updated according to the rule in the assignment.
        """
        new_weights = {}

        own_weights = self.get_weight_copies()
        other_weights = other.get_weight_copies()

        for k in own_weights:
            new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

        self.set_weights(new_weights)


class NNModel(torch.nn.Module):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self, in_dim, out_dim, n_hidden_layers=3, hidden_dim=64):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.fa = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fa(x)


def format_batch(batch, target_network, gamma):
    """
    Input : 
        - batch, a list of n=batch_size elements from the replay buffer
        - target_network, the target network to compute the one-step lookahead target
        - gamma, the discount factor

    Returns :
        - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
        - (actions, targets) : where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.
    """
    # TODO: Implement
    pass


def dqn_loss(y_pred, y_target):
    """
    Input :
        - y_pred, (batch_size, n_actions) Tensor outputted by the network
        - y_target = (actions, targets), where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.

    Returns :
        - The DQN loss 
    """
    # TODO: Implement
    pass


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # NEW


# NEW : Added lr argument
def main(batch_size, gamma, buffer_size, seed, tau, training_interval, lr):
    environment = gym.make("LunarLander-v2")
    set_random_seed(environment, seed)

    actions = list(range(environment.action_space.n))
    model = NNModel(environment.observation_space.shape[0], environment.action_space.n)
    policy_net = DQN(
        actions,
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        loss_function=dqn_loss,
    )
    # NEW: pass a deep copy of the model
    target_net = DQN(actions, deepcopy(model), optimizer="sgd", loss_function=dqn_loss,)
    replay_buffer = ReplayBuffer(buffer_size)

    training_done = False
    episodes_done = 0
    steps_done = 0
    epsilon = 1.0

    while not training_done:
        s = environment.reset()
        episode_done = False
        while not episode_done:
            a = policy_net.get_action(s, epsilon)
            next_s, r, episode_done, _ = environment.step(a)
            replay_buffer.store((s, a, r, next_s, episode_done))
            s = next_s
            steps_done += 1

            if steps_done % training_interval == 0:
                if len(replay_buffer.data) >= batch_size:
                    batch = replay_buffer.get_batch(batch_size)
                    x, y = format_batch(batch, target_net, gamma)
                    loss = policy_net.train_on_batch(x, y)
                    target_net.soft_update(policy_net, tau)

        # TODO: update epsilon
        episodes_done += 1


if __name__ == "__main__":
    """
    All hyperparameter values and overall code structure are
    only given as a baseline. 
    
    You can use them if they help  you, but feel free to implement
    from scratch the required algorithms if you wish !
    """
    batch_size = 32
    gamma = 0.99
    buffer_size = 1e5
    seed = 42
    tau = 1e-2
    training_interval = 4
    lr = 5e-4  # NEW lr as parameter

    # NEW : pass lr to main()
    main(batch_size, gamma, buffer_size, seed, tau, training_interval, lr)
