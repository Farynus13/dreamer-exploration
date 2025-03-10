import gym
import numpy as np
from main import MoveDrones
from map_tools import perlin_noise_2Dmask,generate_maze
import os
from visualisations import create_gif_from_images

class Drone(gym.Env):

  def __init__(self, task, id, size=(64, 64),seed=42, max_act=200, num_drones=3):
    self.id = id
    self._z_dim = 2
    self._map_scale = 1
    self._size = size
    self._window_size = [size[0], size[1], self._z_dim]
    self._map_size = [size[0]*self._map_scale, size[1]*self._map_scale, self._z_dim]

    self.lidar_range = 10
    self._lidar_params={"num_h_rays":32,
                               "num_v_rays":32,
                               "fov_v":180}

    self._max_length =  8 # max num of steps performed after waypoint was generated
    self._task = task
    self._done = False
    self._num_act = 0
    self._max_act = max_act
    self._num_drones = num_drones

    self._previous_voxel_map = None
    self._agents = None
    self._start_position = None

    self._max_distance = self._max_act * self._max_length * self._num_drones * 2
    self._episode_distance = 0
    self._episode_idx = 0
    self._episode_interval = 10
    self._log_episode = False
    self._episode_path = f"gifs/episode{self._episode_idx}/"

    self._maps_path = 'maps/'
    self._maps = os.listdir(self._maps_path)

  def reset(self):
    if self._episode_idx > 0:
      print(f"Episode {self._episode_idx} | Length: {self._num_act} | Exploration: {self.sum_current/self.max_voxels * 100:.2f}% | Distance: {self._episode_distance}")
      if self._log_episode:
        create_gif_from_images(self._episode_path, os.path.join(self._episode_path, 'replay.gif'), self._max_act ,delete_img=True)
        
    self._episode_idx += 1

    self._episode_path = f"gifs/episode{self._episode_idx}/"
    if  self.id==0 and self._episode_idx % self._episode_interval == 0:
      try:
        os.makedirs(self._episode_path)
        self._log_episode  = True
      except FileExistsError:
        self._log_episode  = False
    else:
      self._log_episode  = False

    map_idx = np.random.randint(len(self._maps))
    voxel_map = np.load(os.path.join(self._maps_path, self._maps[map_idx]))

    spawn_points = np.argwhere(voxel_map == 0) 
    np.random.shuffle(spawn_points)
    self._start_position = [position for position in spawn_points[:self._num_drones]]

    self._agents = MoveDrones(voxel_map, self._start_position, self.lidar_range,self._max_length, self._size,\
                              log_images=self._log_episode ,img_path=self._episode_path, lidar_params=self._lidar_params)

    
    self._previous_voxel_map = self._agents.current_voxel_map.copy()
    self._num_act = 0
    self._episode_distance = 0
    self.max_voxels = np.sum(self._agents.gt_surface_map)
    self.sum_current = np.sum(self._agents.current_voxel_map)
    self._done = False
    return self._obs(is_first=True)

  @property
  def observation_space(self):
      return gym.spaces.Dict(
          {
              "image": gym.spaces.Box(0, 255, self._size + (self._z_dim*self._num_drones,), dtype=np.uint8),
              # "image": gym.spaces.Box(0, 255, self._map_size, dtype=np.uint8),
              "is_first": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
              "is_last": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
              "is_terminal": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
              "position": gym.spaces.Box(0,1,(3*self._num_drones,), dtype=np.float32),#"log_player_pos": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32
          }
      )

  @property
  def action_space(self):
    return gym.spaces.Box(0.0, 1.0, (3*self._num_drones,), dtype=np.float32)
  
  def _obs(self, is_first=False, is_last=False, is_terminal=False):
    image = None

    views = self._agents.get_drones_view(self._window_size) 

    image = None
    for i in range(self._num_drones):
      if image is None:
        image = views[i]
      else:
        image = np.concatenate((image, views[i]), axis=2)

    # image = self._agents.current_voxel_map.copy()*255
    positions = self._agents.get_positions()
    # for i in range(self._num_drones):
      # x,y,z = positions[i]
      # image[x,y,z] = 128

    return dict(
        image = image,
        position = self.normalise_position(positions).flatten(),
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal)
    )

  def step(self, action):
    self._num_act += 1

    reward,total_num_steps_performed = self.perform_navigation(action) # here we perform the navigation to the waypoint, and we return the reward
    self._previous_voxel_map = self._agents.current_voxel_map.copy()

    self._done = (self._num_act >= self._max_act or \
              self.sum_current >= 0.99 * self.max_voxels
              )
    
    return self._obs(is_last=self._done, is_terminal=self._done), np.float32(reward),self._done, {}


  def normalise_position(self,position):
    normalised_position = np.zeros_like(position, dtype=np.float32)
    for i in range(3):
      normalised_position[:,i] = position[:,i] / self._map_size[i]
    return normalised_position

  
  def perform_navigation(self, action):
    waypoints = self._agents.action_to_waypoint(action)
    total_num_steps_performed = self._agents.move_all_drones(waypoints)
    self._episode_distance += np.sum(total_num_steps_performed)
    # steps_per_drone = total_num_steps_performed / self._num_drones
    return self.calculate_reward(total_num_steps_performed),total_num_steps_performed
  
  def calculate_reward(self, total_num_steps_performed):
    # Total voxels explored in the previous and current maps
    sum_previous = np.sum(self._previous_voxel_map)
    self.sum_current = np.sum(self._agents.current_voxel_map)

    # Compute the number of newly explored voxels
    delta_explored = self.sum_current - sum_previous

    # Total number of steps performed by all agents in this timestep
    sum_num_steps = np.sum(total_num_steps_performed)

    # Hyperparameters for reward scaling
    alpha = 100.0  # Weight for exploration reward
    c_mult = 2
    beta = 0.05     # Weight for efficiency penalty
    c_steps = 1.0    # Penalty per step
    c_inference = 5.0  # Penalty for an inference (adjust as needed)
    bonus = 100.0    # Large bonus for completing exploration

    # Calculate exploration reward
    exploration_mult = 1 + sum_previous / self.max_voxels * c_mult
    exploration_reward = (delta_explored / self.max_voxels) * alpha * exploration_mult

    # Calculate efficiency penalty
    efficiency_penalty = -beta * (c_steps * sum_num_steps + c_inference)

    # Completion bonus
    completion_bonus = bonus if self.sum_current >= 0.99 * self.max_voxels else 0.0

    # Total reward
    reward = exploration_reward + efficiency_penalty + completion_bonus

    return reward

