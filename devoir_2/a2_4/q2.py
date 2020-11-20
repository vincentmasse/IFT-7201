import numpy as np
import gym
from tilecoding import TileCoder


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


def sarsa_linear_fa(tile_coder, theta, state, actions):
    phis_indices = np.array([tile_coder.phi(state, action) for action in actions])
    actions_vals = []
    print(state.shape)
    for tiles_indices in phis_indices:
        actions_vals.append(sum(np.take(theta, tiles_indices)))

    return actions_vals


def choose_action(q_vals):
    return np.argmax(q_vals)


def get_targets(next_q_vals, next_action, reward, gamma):
    target = reward + gamma + next_q_vals[next_action]

    return target


def update_theta(tile_coder, theta, state, action, target, q_vals, lr):
    predicted = q_vals[action]
    diff = target - predicted
    phi = sum(np.take(theta, tile_coder.phi(state, action)))

    grad = -2 * diff * phi
    theta -= lr * grad


def sarsa_algorithm(environment, tile_coder, theta, n_episodes=300, max_steps=200, gamma=1, learning_rate=0.1):
    actions = list(range(environment.action_space.n))

    for n_episode in range(n_episodes):
        state = environment.reset()
        action = choose_action(sarsa_linear_fa(tile_coder, theta, state, actions))
        G = 0

        for step in range(max_steps):
            next_state, r, done, _ = environment.step(action)
            if done:
                break
            G += r
            q_vals = sarsa_linear_fa(tile_coder, theta, next_state, actions)
            next_action = choose_action(q_vals)

            target = get_targets(q_vals, next_action, r, gamma)
            update_theta(tile_coder, theta, state, action, target, q_vals, learning_rate)

            state = next_state
            action = next_action

        if n_episode % 10 == 0:
            print(f'After {n_episode} episode, we have G_0 = {G:.2f}')

    return theta


def main(seed):
    environment = gym.make("MountainCar-v0")

    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)
    theta = np.zeros(tile_coder.size)

    sarsa_algorithm(environment, tile_coder, theta)

if __name__ == "__main__":
    seed = 42
    main(seed)
