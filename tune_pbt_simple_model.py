import numpy as np
import ray
import ray.tune as tune
from ray.tune import CLIReporter

from trainable_simple import SimpleTrainer

def main():

    ray.init(address="auto")
    reporter = CLIReporter()
    reporter.add_metric_column("h")
    reporter.add_metric_column("theta")

    analysis = tune.run(
        SimpleTrainer,
        stop={"training_iteration": 10},
        config={
            "theta": np.array([0.9, 0.9]),
            "h": np.array([1.,0.]),
            "alpha": 0.01
        },
        progress_reporter=reporter
    )
    print('best config: ', analysis.get_best_config(metric="score", mode="max"))
    best_config = analysis.get_best_config(metric="score", mode="max")
    print(best_config)


if __name__ == "__main__":
    main()