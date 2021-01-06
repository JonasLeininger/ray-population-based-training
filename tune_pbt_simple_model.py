import ray
import ray.tune as tune
from ray.tune import CLIReporter

from trainable_simple import SimpleTrainer

def main():

    ray.init(address="auto")

    analysis = tune.run(
        SimpleTrainer,
        stop={"training_iteration": 20},
        config={
            "a": 2,
            "b": 4
        }
    )
    print('best config: ', analysis.get_best_config(metric="score", mode="max"))


if __name__ == "__main__":
    main()