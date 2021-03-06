import ray
import ray.tune as tune
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from config.config import Config
from training.trainer_reacher import ReacherTrainer


def main():
    config = Config(config_file="config/config_jonas_local.yaml").config

    ray.init(address="auto")

    space = {
        "agent_learn_every_x_steps": hp.choice("agent_learn_every_x_steps", [10, 20]),
        "replay_buffer_batch_size": hp.choice("replay_buffer_batch_size", [128, 512, 1024]),
        # "memory_learning_start": hp.choice("memory_learning_start", [50000]),
        # "agent_gamma": hp.uniform("agent_gamma", 0.95, 0.999)
        "agent_gamma": hp.choice("agent_gamma", [0.95, 0.995])
    }

    reporter = CLIReporter()
    reporter.add_metric_column("mean_of_mean")
    reporter.add_metric_column("score")

    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_of_mean",
        mode="max",
        grace_period=500,
        max_t=3600)

    tune.run(
        ReacherTrainer,
        name="asynHyber-reacher-ddpg",
        scheduler=ahb,
        config=config,
        queue_trials=True,
        num_samples=10,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 4, "gpu": 0.2},
        search_alg=HyperOptSearch(
            space=space, max_concurrent=3, metric="mean_of_mean", mode="max"
        ),
        checkpoint_freq=20,
        checkpoint_at_end=True,
        verbose=1,
    )


if __name__ == "__main__":
    main()