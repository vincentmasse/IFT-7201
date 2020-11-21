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

"""
def get_expected_reward(tile_coder, theta, state, action, traces):
    feature_indices = tile_coder.phi(state, action)
    estimation = np.sum(theta[feature_indices])

    traces[feature_indices] = 1

    return estimation
"""


def sarsa_linear_fa(tile_coder, theta, state, actions):
    phis_indices = np.array([tile_coder.phi(state, action) for action in actions])
    actions_vals = []

    for tiles_indices in phis_indices:
        actions_vals.append(np.sum(theta[tiles_indices]))

    return actions_vals, phis_indices


def choose_action(q_vals):
    return np.argmax(q_vals)


def get_delta(reward, q_vals, action, next_q_vals, next_action, gamma):
    delta = reward + (gamma * next_q_vals[next_action]) - q_vals[action]

    return delta


def update_traces(traces, gamma, lmbda):
    traces *= gamma * lmbda


def update_theta(theta, estimation, target, traces, lr):
    delta = target - estimation
    theta += lr * delta * traces


def replace_traces(action, features, traces):
    traces[features[action]] = 1


def sarsa_algorithm(environment, tile_coder, n_episodes=300, max_steps=200, gamma=1, lmbda=0.9, learning_rate=0.1):
    theta = np.zeros(tile_coder.size)
    actions = list(range(environment.action_space.n))
    G_episode_cum = 0

    for n_episode in range(n_episodes):
        traces = np.zeros(tile_coder.size)
        G = 0

        state = environment.reset()
        for step in range(max_steps):
            q_vals, features = sarsa_linear_fa(tile_coder, theta, state, actions)
            action = choose_action(q_vals)
            replace_traces(action, features, traces)

            next_state, r, done, _ = environment.step(action)

            G += r

            # delta = r - q_vals[action]

            if done:
                target = r
                update_theta(theta, q_vals[action], target, traces, learning_rate)
                break

            next_q_vals, _ = sarsa_linear_fa(tile_coder, theta, next_state, actions)
            next_action = choose_action(next_q_vals)

            target = r + gamma * next_q_vals[next_action]
            update_theta(theta, q_vals[action], target, traces, learning_rate)
            update_traces(traces, gamma, lmbda)

            state = next_state

        print(f'After {n_episode} episode, we have G_0 = {G:.2f}')
        G_episode_cum += G
        if n_episode % 100 == 0:
            print(f'--------CUMULATIVE G MEAN : {G_episode_cum/100:.2f}--------')
            G_episode_cum = 0


    return theta


def test_model(theta):
    environment = gym.make("MountainCar-v0")
    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)
    env = gym.wrappers.Monitor(environment, "demos", force=True)

    actions = list(range(environment.action_space.n))
    done = False

    s = environment.reset()
    while not done:
        env.render()
        q_vals, _ = sarsa_linear_fa(tile_coder, theta, s, actions)
        action = np.argmax(q_vals)
        next_s, r, done, _ = environment.step(action)
        s = next_s
    env.close()


def main(seed):
    environment = gym.make("MountainCar-v0")

    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)

    theta = sarsa_algorithm(environment, tile_coder)
    test_model(theta)


if __name__ == "__main__":
    seed = 42
    main(seed)
