import numpy as np

from trainable_simple import SimpleTrainer

def main():
    config1 = {
        "trainer_id": 0,
        "theta": np.array([0.9, 0.9]),
        "h": np.array([1., 0.]),
        "alpha": 0.01,
        "exploit": True,
        "explore": True
    }
    config2 = {
        "trainer_id": 1,
        "theta": np.array([0.9, 0.9]),
        "h": np.array([0., 1.]),
        "alpha": 0.01,
        "exploit": True,
        "explore": True
    }
    l_config = [config1, config2]

    result_trainers = run(200, config_list=l_config)

def run(steps, config_list, explore=True, exploit=True):
    l_scores = []
    l_parameters = []
    arr_score = np.zeros((2, 1))
    arr_thetas = np.zeros((2, 2))

    trainers = [
        SimpleTrainer(config=config_list[0]),
        SimpleTrainer(config=config_list[1])
        ]

    for step in range(steps):
        print(step)
        for trainer in trainers:
            result = trainer.step()
            arr_score[result["id"]] = np.copy(result["score"])
            arr_thetas[result["id"]] = np.copy(result["theta"])
        
        l_scores.append(arr_score)
        l_parameters.append(arr_thetas)

        best_trainer_id = np.argmax(arr_score)
        print("best trainer id", best_trainer_id)
        print("scores", arr_score)
        best_params = np.copy(arr_thetas[best_trainer_id])
        print("thetas ", arr_thetas)
        print("best thetas ", best_params)

        if step % 10 == 0 and step > 0:
            for trainer in trainers:
                if explore and exploit:
                    bool_explore = trainer.exploit(best_trainer_id, best_params)
                    if bool_explore:
                        trainer.explore()
                elif explore and not exploit:
                    trainer.explore()
                elif not explore and exploit:
                    trainer.exploit()
                else:
                    pass
    
    return trainers, l_scores, l_parameters


if __name__ == "__main__":
    main()