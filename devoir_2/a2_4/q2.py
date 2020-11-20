import numpy as np
import gym
from tilecoding import TileCoder
from replayBuffer import ReplayBuffer


def get_tile_coder(environment):
    return TileCoder(
        environment.observation_space.high,
        environment.observation_space.low,
        num_tilings=8,
        tiling_dim=8,
        max_size=4096,
    )


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)


def linear_fa(tile_coder, theta, state, actions):
    """
        Returns the estimated q-values of each (s,a) pair according to the dot-product <theta, phi(s,a)>.
    """
    print('-->', len(state))
    phis = np.array([tile_coder.phi(state, action) for action in actions])


    print(phis)
    print(theta.shape)
    action_vals = phis.dot(theta)

    return action_vals

def sarsa_linear_fa(tile_coder, theta, state, actions):
    phis_indices = np.array([tile_coder.phi(state, action) for action in actions])
    phis = []

    for tiles_indices in phis_indices:
        phis.append(np.take(theta, tiles_indices))
    #actions_vals = theta[0] +

def choose_action(q_vals):
    return np.argmax(q_vals)

def get_targets(next_q_vals, rewards, gamma):
    pass


def sarsa_algorithm(environment, tile_coder, theta, n_episodes=300, buffer_size=50000, batch_size=32, max_steps=200, gamma=1, learning_rate=0.1):
    actions = list(range(environment.action_space.n))
    replay_buffer = ReplayBuffer(buffer_size)

    for n_episode in range(n_episodes):
        state = environment.reset()
        action = choose_action(linear_fa(tile_coder, theta, state, actions))
        G = 0

        for step in range(max_steps):
            next_state, r, done, _ = environment.step(action)
            if done:
                break
            q_vals = linear_fa(tile_coder, theta, next_state, actions)
            next_action = choose_action(q_vals)

            G += r
            replay_buffer.store((state, action, r, next_state, next_action))

            state = next_state
            action = next_action

            if len(replay_buffer.data) > batch_size:
                minibatch = replay_buffer.get_batch(batch_size)

                states = np.array([x[0] for x in minibatch])
                actions_taken = np.array([x[1] for x in minibatch])
                rewards = np.array([x[2] for x in minibatch])
                next_states = np.array([x[3] for x in minibatch])
                next_actions = np.array([x[4] for x in minibatch])

                next_q_values_predicted = np.array([linear_fa(tile_coder, theta, s, actions) for s in next_states])
                print(next_q_values_predicted)







def main(seed):
    environment = gym.make("MountainCar-v0")

    set_random_seed(environment, seed)
    print(environment.observation_space.shape)
    tile_coder = get_tile_coder(environment)
    theta = np.zeros(tile_coder.size)

    sarsa_algorithm(environment, tile_coder, theta)


if __name__ == "__main__":
    seed = 42
    main(seed)
