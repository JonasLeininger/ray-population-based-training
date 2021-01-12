import numpy as np
import torch

from agent.ornstein_uhlenbeck_noise import Noise
from model.actor import DDPGActor
from model.critic import DDPGCritic

class BipedalAgent():

    def __init__(self, config):
        self.actor_local = DDPGActor(config)
        self.actor_target = DDPGActor(config)
        self.optimizer_actor = torch.optim.Adam(
            self.actor_local.parameters(), lr=config["learning_rate"]
        )
        self.critic_local = DDPGCritic(config)
        self.critic_target = DDPGCritic(config)
        self.optimizer_critic = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["optimizer_critic_weight_decay"],
        )

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
