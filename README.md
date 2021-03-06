# ray-population-based-training
Tutorial for Population Based Training, PBT, with ray[tune], ray[rllib] and PyTorch

Paper on PBT by Deepmind https://arxiv.org/abs/1711.09846

## Python Environment
Create environment
```sh
python3 -m venv ~/envs/ray-pbt
```
Activate environment
```sh
source ~/envs/ray-pbt/bin/activate
```
Deactivate environment
```sh
deactivate
```
### Install Requirements

Install the pytorch version dependent on your cuda setup. For a cuda 10.1 install:
```sh
pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
#### Install Gym

```sh
git clone git@github.com:openai/gym.git
cd gym
pip install -e .
```
```sh
git clone git@github.com:pybox2d/pybox2d.git
cd pybox2d
pip install -e '.[box2d]'
```

#### Install Requirements
install requirements.txt
```sh
pip install -r requirements.txt
```

## Ray local server
Start local ray server
```sh
ray start --head
```
Stop ray server
```sh
ray stop
```