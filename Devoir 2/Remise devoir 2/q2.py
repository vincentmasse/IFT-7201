import numpy as np
from matplotlib import pyplot
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


def choose_action(q_vals):
    return np.argmax(q_vals)


def linear_af(tile_coder, theta, state, actions):
    phis_indices = np.array([tile_coder.phi(state, action) for action in actions])
    actions_vals = []

    for tiles_indices in phis_indices:
        actions_vals.append(np.sum(theta[tiles_indices]))

    return actions_vals, phis_indices


def get_delta(reward, q_vals, action, next_q_vals, next_action, gamma):
    delta = reward + (gamma * next_q_vals[next_action]) - q_vals[action]

    return delta


def replace_traces(action, features, traces):
    traces[features[action]] = 1


def update_traces(traces, gamma, lmbda):
    traces *= gamma * lmbda


def update_theta(theta, estimation, target, traces, lr):
    delta = target - estimation
    theta += lr * delta * traces


def sarsa_delta_algorithm(environment, tile_coder, n_episodes, max_steps, gamma, lmbda, alpha):
    theta = np.zeros(tile_coder.size)
    actions = list(range(environment.action_space.n))

    for n_episode in range(n_episodes):
        traces = np.zeros(tile_coder.size)
        G = []

        state = environment.reset()
        for step in range(max_steps):
            q_vals, features = linear_af(tile_coder, theta, state, actions)
            action = choose_action(q_vals)
            replace_traces(action, features, traces)

            next_state, r, done, _ = environment.step(action)

            G.append(r)

            if done:
                target = r
                update_theta(theta, q_vals[action], target, traces, alpha)
                break

            next_q_vals, _ = linear_af(tile_coder, theta, next_state, actions)
            next_action = choose_action(next_q_vals)

            target = r + gamma * next_q_vals[next_action]
            update_theta(theta, q_vals[action], target, traces, alpha)
            update_traces(traces, gamma, lmbda)

            state = next_state

        yield n_episode, theta, G


def test_model(theta, seed):
    environment = gym.make("MountainCar-v0")
    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)
    env = gym.wrappers.Monitor(environment, "demos", force=True)

    actions = list(range(environment.action_space.n))
    done = False

    s = environment.reset()
    while not done:
        env.render()
        q_vals, _ = linear_af(tile_coder, theta, s, actions)
        action = np.argmax(q_vals)
        next_s, r, done, _ = environment.step(action)
        s = next_s
    env.close()


def sarsa_experiment(
        seed=None,
        n_episodes=500,
        max_steps=200,
        gamma=1,
        lmbda=0.9,
        alpha=0.1,
        test=False,
        plot=False
):
    environment = gym.make("MountainCar-v0")

    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)
    theta = []
    G_MEAN_100 = []
    G_acc = []

    print(f"\n--------SARSA(λ) with: λ = {lmbda} γ = {gamma} α = {alpha}--------\n")

    for n_episode, newTheta, G in sarsa_delta_algorithm(
        environment,
        tile_coder,
        n_episodes,
        max_steps,
        gamma,
        lmbda,
        alpha
    ):
        theta = newTheta
        g_sum = sum(G)
        G_acc.append(g_sum)
        G_MEAN_100.append(np.mean(G_acc[-100:]))
        if n_episode % 100 == 0:
            print(f'-----G MEAN : {G_MEAN_100[-1:][0]:.2f}-----')
        if n_episode % 10 == 0:
            print(f'After {n_episode} episode, we have G = {g_sum:.2f}')

    if plot:
        pyplot.plot(G_MEAN_100, label=f"SARSA(λ) with: λ = {lmbda} γ = {gamma} α = {alpha}")
        pyplot.xlabel("Episodes")
        pyplot.ylabel("Last 100 episodes gains mean")
        pyplot.title(f"SARSA(λ) experiments")
        pyplot.legend()
    if test:
        test_model(theta, seed)


def main():
    SEED = 42
    sarsa_experiment(seed=SEED, plot=True, lmbda=0)
    sarsa_experiment(seed=SEED, plot=True, lmbda=0.5)
    sarsa_experiment(seed=SEED, plot=True, lmbda=0.9)
    sarsa_experiment(seed=SEED, plot=True, lmbda=1)

    pyplot.show()


if __name__ == "__main__":
    main()
