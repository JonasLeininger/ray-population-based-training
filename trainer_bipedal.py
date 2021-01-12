import numpy as np
from ray import tune
import gym

from utils.annotations import override
from config.config import Config
from agent.agent_bipedal import  BipedalAgent
from model.prioritized_experience_replay_buffer import PrioritizedExperienceReplay

class BipedalTrainer(tune.Trainable):
    _name = "BipedalTrainer"

    @override(tune.Trainable)
    def setup(self, config):
        self.env = gym.make('BipedalWalker-v3')
        self.agent = BipedalAgent(config)
        self.obs = self.env.reset()
        self.per = PrioritizedExperienceReplay(
            capacity=config["replay_buffer_memory_size"]
            )
        self.dones = np.array(0)
        self.experience_count = 0
        self.train_count = 0
        self.score = 0
    
    @override(tune.Trainable)
    def step(self):
        self._execute_iteration()
        return {"reward": self.score, "done": self.dones}
    
    def _execute_iteration(self):
        self._reset_env()
        self._run_environment_episode()
        self.train_count += 1

    def _reset_env(self):
        self.obs = self.env.reset()
        self.obs = np.expand_dims(self.obs, axis=0)
        self.timestep_count_env = 0
        self.dones = np.array(0)
        self.score = 0

    def _run_environment_episode(self):
        while not np.any(self.dones.astype(dtype=bool)):
            self.experience_count += 1
            self.env.render(mode='rgb_array')
            actions = self.agent.act(self.obs)
            next_obs, rewards, dones, info = self.env.step(actions.flatten())
            self.score += rewards

            # actions = np.expand_dims(actions, axis=0)
            rewards = np.array((rewards,))
            rewards = np.expand_dims(rewards, axis=0)
            next_obs = np.expand_dims(next_obs, axis=0)
            dones = np.array((dones,))
            dones = np.expand_dims(dones, axis=0)
            experience = self.obs, actions, rewards, next_obs, dones
            self.per.store(experience)
            self._learn(self.timestep_count_env)
            self.dones = dones
            self.obs = next_obs
            self.timestep_count_env += 1

    def _learn(self, timestep_count):
        if (self.experience_count >= self.config["memory_learning_start"]) and (
            timestep_count % self.config["agent_learn_every_x_steps"] == 0
        ):
            self._train_model()

    def _train_model(self):
        for learn_iteration in range(self.config["agent_learn_num_iterations"]):
            b_idx, experience, b_ISWeights = self.per.sample(
                self.config["replay_buffer_batch_size"]
            )
            obs, actions, rewards,next_obs, dones = experience
            absolute_errors = self.agent.learn(
                experience, weights=b_ISWeights, weighted_loss=True
            )
            self.per.batch_updates(b_idx, absolute_errors)


if __name__ == "__main__":
    config = Config(config_file="config/config_local.yaml").config
    trainer = BipedalTrainer(config=config)
    for i in range(10):
        result_dict = trainer.step()
        print(result_dict)