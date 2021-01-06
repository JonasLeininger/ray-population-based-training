from ray import tune

from annotations import override

def objective(x, a, b):
    return a * (x**0.5) + b


class SimpleTrainer(tune.Trainable):
    _name = "SimpleTrainer"

    @override(tune.Trainable)
    def setup(self, config):
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]
    
    @override(tune.Trainable)
    def step(self):
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}
