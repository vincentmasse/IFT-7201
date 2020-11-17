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


def main(seed):
    environment = gym.make("MountainCar-v0")
    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)
    theta = np.zeros(tile_coder.size)

    # Example how to use the tile coder
    s = environment.reset()
    action = 0
    x = tile_coder.phi(s, action)


if __name__ == "__main__":
    seed = 42

    main(seed)
