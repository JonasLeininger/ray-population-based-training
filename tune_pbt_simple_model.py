import ray
import ray.tune as tune
from ray.tune import CLIReporter

def main():

    ray.init(address="auto")
    reporter = CLIReporter()


if __name__ == "__main__":
    main()