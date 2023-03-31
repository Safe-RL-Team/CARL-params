# Caution Parameters for Reinforcement Learning in Safety-Critical Settings (CARL).

<figure>
  <p style="text-align:center;">
  
  <img src="/README_assets/CARLreward.jpg" alt="Figure1" class = "center" width = "200" height = "200">
  <img src="/README_assets/CARLstate.jpg" alt="Figure2" class = "center" width = "200" height = "200">
  </p>
</figure>


This project was created as part of the course “Advanced Topics in Reinforcement Learning” at TU Berlin in the WS22/23. The project is outlined in [my blogpost](https://Safe-RL-Team.github.io/CARL/caution-params/). This repository is a slightly adapted version of the [original code](https://github.com/jesbu1/carl) for the work of Zhang et al. (2020): [Cautious Adaptation for RL in Safety-Critical Settings (CARL)](https://arxiv.org/abs/2008.06622). CPU multiprocessing is enabled.


## Installation
Clone this repository with `git clone https://github.com/Safe-RL-Team/CARL_params.git`.
In order to experiment on MuJoco environments, you must have MuJoco 200 installed with an appropriate MuJuco license linked.
See here to download and setup MuJoco 200: [mujoco](https://www.roboti.us/index.html). On Ubuntu, we had to install some extra packages first: `sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`.

To install the base packages required, simply `pip install -r requirements.txt` (Tested with Python 3.6). 


## Running for Experiments with Different Caution Parameters
Experiments for CARL State or CARL Reward over a range of caution parameters and target domains can be run using the necessary flags:

```
python exp_caution_params.py --CARL 'CARL' --min_caution 'min_caution' --max_caution 'max_caution' --ncaution_params 'ncaution_params' 
```

Here is an example for running CARL State, looping over caution parameter lambda_2 in {0.5, 1, 1.5, 2} and target domains with pole length {1, 2} and using a pretrained model from 'log/ex_dir'.

```
python exp_caution_params.py --CARL State --min_caution 0.5 --max_caution 2 --ncaution_params 4 --min_td 1 --max_td 2 --ntds 2 --pretrain_dir log/ex_dir
```

Results will be saved in `log/<date+time of experiment start>_<caution_param>_td_<test_domain>/`.
Trial data will be contained in `log/<date+time of experiment start>_<caution_param>_td_<test_domain>/-tboard`, 
You can run `tensorboard --logdir <logdir>` to visualize the results.

## Directory Structure
Configuration files are located in `config/`, modify these python files to change some environment/model/training parameters.

The ensemble model class is also located in `config/`.

`env/` contains gym environment files.

`MPC.py` contains training code and acting code for the MPC controller.

`optimizers.py` contains the optimizers (CEM) used for optimizing actions with MPC.

`MBExperiment.py` contains training, adaptation, and testing loop code.

`Agent.py` contains logic for interacting with the environment with model-based planning and collecting samples
