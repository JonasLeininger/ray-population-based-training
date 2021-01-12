import numpy as np

from agent.ornstein_uhlenbeck_noise import Noise

class BipedalAgent():

    def __init__(self, config):
        self.noise = Noise(
            config["action_dim"],
            np.asarray(config["noise_mu"]),
            theta=config["noise_theta"],
            sigma=config["noise_sigma"],
        )
        self.gamma = config["agent_gamma"]
        self.tau = config["agent_tau"]
        self.device = config["device"]

    def act(self, obs):
        actions = np.random.random((4,))
        return actions
    
    def learn(self, exp, weights=None, weighted_loss=False):
        return 0
