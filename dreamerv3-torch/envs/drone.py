import gym
import numpy as np
from main import MoveDrones
from map_tools import perlin_noise_2Dmask,generate_maze
import os
from visualisations import create_gif_from_images

class Drone(gym.Env):

  def __init__(self, task, action_repeat, size=(64, 64),seed=42, max_act=120, num_drones=3):
    self._z_dim = 2
    self._map_scale = 1
    self._size = size
    self._window_size = [size[0], size[1], self._z_dim]
    self._map_size = [size[0]*self._map_scale, size[1]*self._map_scale, self._z_dim]

    self.lidar_range = 6
    self._lidar_params={"num_h_rays":32,
                               "num_v_rays":32,
                               "fov_v":120}

    self._max_length =  8 # max num of steps performed after waypoint was generated
    self._task = task
    self._done = False
    self._num_act = 0
    self._max_act = max_act
    self._num_drones = num_drones

    self._previous_voxel_map = None
    self._agents = None
    self._start_position = None

    self._episode_distance = 0
    self._episode_idx = 0
    self._episode_interval = 6
    self._log_episode = False
    self._episode_path = f"gifs/episode{self._episode_idx}/"

    self._maps_path = 'maps/'
    self._maps = os.listdir(self._maps_path)

  def reset(self):

    self._episode_idx += 1

    self._episode_path = f"gifs/episode{self._episode_idx}/"
    if  self._episode_idx % self._episode_interval == 0 and not os.path.exists(self._episode_path):
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
    self._done = False
    return self._obs(is_first=True)

  @property
  def observation_space(self):
      return gym.spaces.Dict(
          {
              "image": gym.spaces.Box(0, 255, self._size + (self._z_dim*self._num_drones,), dtype=np.uint8),
              "is_first": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8),
              "is_last": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8),
              "is_terminal": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8),
              "position": gym.spaces.Box(0,1,(3*self._num_drones,), dtype=np.float32)#"log_player_pos": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32
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

    return dict(
        image = image,
        position = self.normalise_position(self._agents.get_positions()).flatten(),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

  def step(self, action):
    # print("Step: ", self._num_act)
    if self._done:
      #print episode max_act, % of explored voxels
      print(f"Episode {self._episode_idx} | Length: {self._num_act} | Exploration: {np.sum(self._agents.current_voxel_map) / np.sum(self._agents.gt_surface_map) * 100:.2f}% | Distance: {self._episode_distance}")
      if self._log_episode:
        create_gif_from_images(self._episode_path, os.path.join(self._episode_path, 'replay.gif'), self._max_act ,delete_img=True)
      # total_explo = np.sum(self._agents.current_voxel_map)/ np.sum(self._agents.gt_voxel_map) * 100
      # print("Episode efficiency: ", total_explo / self._episode_distance)
      return self.reset(),0,True, {}
    self._num_act += 1

    reward,total_num_steps_performed = self.perform_navigation(action) # here we perform the navigation to the waypoint, and we return the reward
    self._previous_voxel_map = self._agents.current_voxel_map.copy()

    self._done = (self._num_act >= self._max_act or \
              np.sum(self._agents.current_voxel_map) >= 0.99 * np.sum(self._agents.gt_surface_map)
    )
    
    return self._obs(is_last=self._done, is_terminal=self._done), np.float32(reward),False, {}


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
    sum_previous = np.sum(self._previous_voxel_map)
    sum_current = np.sum(self._agents.current_voxel_map)
    max_voxels = np.sum(self._agents.gt_surface_map)
    sum_num_steps = np.sum(total_num_steps_performed)

    #we want to incentivize percent of newly explored voxels as well as penalize number of steps
    new_voxels = sum_current - sum_previous

    a = 4
    b = 0.1
    c = 2
    d = (self._max_length*2*self._num_drones+1)*self._max_act
    decay_factor = max(c*(1 - (self._num_act+self._episode_distance)/d),0)
    new_voxel_factor = a * sum_current / max_voxels + 1  
    r_new = new_voxels/max_voxels * new_voxel_factor*100 * decay_factor
    r_num_acts = b * (sum_num_steps + 1) 
    
    reward = r_new - r_num_acts
    return reward
