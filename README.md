# Multiagent Exploration Framework with Dreamer-v3 based Reinforcement Learning

This project aims to develop a framework for 3D lidar-based Multiagent Exploration Planning with Dreamer-v3 RL model. The goal is to obtain scalable and robust centralised planner that will generate waypoints for all agents to perform optimal exploration of the 3d environment.

### Project Overview
By stating the multiagent exploration in terms of optimization problem where one has to optimize the distance/time required for full exploration of the unknown 3d environment, we can harness the power of reinforcement learning methods in learning the optimal policy. Specifically we use Dreamer-v3 model developed by Google DeepMind to learn optimal waypoint generation using efficient lidar based simulation of our design.

![Figure 1: Exploration](/media/exploration.gif)

*Gif shows how 3 agents learn to efficiently explore 3d environment (although it looks like the environment is a 2d one,
 in reality the dimension of environment is 64x64x2 voxels. We reduced the z dimmension in order to efficiently iterate
  framework developement but it's fully adapted to handle varying size 3d environments)*

### Framework Description
Current version of the framework utilizes Dreamer-v3,<sup>[[1]](https://doi.org/10.48550/arXiv.2301.04104)</sup> (we use [PyTorch implementation](https://github.com/NM512/dreamerv3-torch)) as our policy model. We train the model in the customised gym environment that runs the simulation of the multiagent exploration process, where our policy model generates waypoints for each agent as an action. During the rollout of the action all of the agents perform navigation to the generated waypoints using classical algorithms like A*, where exploration of new voxels is being rendered by the bresenhams algorithm creating a lightweight simulation environment.

Our observation space is the 3d voxel space around our agents stacked together as a one 3D tensor of uint8 values. We also observe states of all agents like their current positions. We modified Dreamer-v3 encoder module to handle 3D inputs efficiently by implementing 3D Sparse Convolution<sup>[[2]](https://doi.org/10.48550/arXiv.1505.02890)</sup>. Vector inputs are handled by the mlp encoder. We can also extend the framework to allow for multimodal exploration using combination of camera and lidar by using camera inputs through the original dreamer's cnn.

### Installation 
Install dreamer-v3 dependencies:
```sh
pip install -r dreamerv3-torch/requirements.txt
```
Install remaining dependencies:
```sh
pip install -r requirements.txt
```

### Training 
1. First prepare maps dataset by running:
```sh
python generate_training_maps.py
  --num_maps 1
  --map_size 64
  --map_scale 1
  --output_dir maps
```
2. To start training run:
```sh
source train.sh path_to_your_logdir 
```
You can resume training by adding resume flag:
```sh
source train.sh logdir_path --resume
```
3. Monitor results:
```sh
tensorboard --logdir ./logdir
```

Example evaluation return from tensorboard:

![Figure 1: Eval Loss](/media/eval_return.png)

### Testing
1. To evaluate the model you can run:
```sh
python benchmark.py 
  --model_path path_to_your_model 
  --exploration_percentage 0.95
```
This script will evaluate model and compare performance with classical benchmark
  - exploration_percentage: when to finish exploration

2. To perform qualitative evaluation you can run:
```sh
python demonstration.py
  --model_path path_to_your_model
```
This script will prepare visualisation of the exploration in the form of a gif (like on Figure 1)

### References

[[1] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap, Mastering Diverse Domains through World Models](https://doi.org/10.48550/arXiv.2301.04104)

[[2] Ben Graham, SParse 3D convolutional neural networks](https://doi.org/10.48550/arXiv.1505.02890)

[[3] Iou-Jen Liu, Unnat Jain, Raymond A. Yeh, Alexander G. Schwing, Cooperative Exploration for Multi-Agent Deep Reinforcement Learning](https://doi.org/10.48550/arXiv.2107.11444)




