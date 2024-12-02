from raycasting import generate_lidar_rays,execute_lidar_rays, visualize_voxel_maps, visualize_lidar_rays,mock_lidar
from navigation_tools import a_star_3d
import numpy as np
import random
from visualisations import visualize_voxel_map
from map_tools import generate_maze, perlin_noise_2Dmask
import matplotlib.pyplot as plt
import os
import matplotlib
import ctypes
#matplotlib.use('Agg')



class MoveDrones:
    def __init__(self, gt_voxel_map, start_positions,lidar_range=100, 
                 max_steps=10,
                 window_size=[64,64],
                 num_drones=2,
                 lidar_params={"num_h_rays":124,
                               "num_v_rays":32,
                               "fov_v":80},
                               log_images=False,
                               img_path="gifs/"):
        """
        Initializes the MoveDrones class with a voxel map and positions for each drone.
        :param voxel_map: 3D numpy array representing the voxel space.
        :param drone_positions: List of tuples representing the starting positions of the drones.
        """
        self.max_steps = max_steps  
        self._num_drones = num_drones
        self._window_size = window_size
        self.gt_voxel_map = gt_voxel_map
        self.current_voxel_map = np.zeros_like(self.gt_voxel_map, dtype=bool)
        self.exploration_map = np.zeros_like(self.gt_voxel_map, dtype=bool)
        self.lidar_range = lidar_range
        
        self.lidar_rays = generate_lidar_rays(lidar_params["num_h_rays"], lidar_params["num_v_rays"], lidar_params["fov_v"], self.lidar_range)
        self.num_rays = self.lidar_rays.shape[0]
        self.lidar_rays_pointer = self.lidar_rays.flatten().astype(ctypes.c_int32) #One could look into not creating a copy and flattening but sending it, like voxel map
        self.lidar_rays_pointer = self.lidar_rays_pointer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        voxel_map_x, voxel_map_y, voxel_map_z = self.gt_voxel_map.shape
        self.gt_voxel_map_pointer = self.gt_voxel_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        assert self.gt_voxel_map_pointer is not None
        assert self.lidar_rays_pointer is not None

        for pos in start_positions:
            execute_lidar_rays(self.gt_voxel_map_pointer, voxel_map_x, voxel_map_y, voxel_map_z, self.exploration_map,pos, self.lidar_range,self.lidar_rays_pointer, self.num_rays)
            # mock_lidar(self.exploration_map,pos,self.lidar_range)
            self.current_voxel_map[:,:,:] = self.exploration_map & self.gt_voxel_map
        self.drones = [
            Drone(self.gt_voxel_map,self.gt_voxel_map_pointer, self.current_voxel_map, self.exploration_map, np.array(pos), id, self.lidar_range, self.lidar_rays_pointer,self.num_rays)
            for id, pos in enumerate(start_positions)]
 
        self.gt_surface_map = VolumeToSurface(self.gt_voxel_map, [np.array(pos) for pos in start_positions])

        self.log_images = log_images
        self.img_path = img_path
        self.img_idx = 0
    
    def get_drones_view(self, window_size=[64,64,3]):
        positions = {id: drone.location.tolist() for id, drone in enumerate(self.drones)}
        submap = {id: np.zeros(window_size, dtype=np.uint8) for id in positions}
        for id in positions:
            x_start = max(positions[id][0] - window_size[0] // 2, 0)
            x_end = min(positions[id][0] + window_size[0] // 2, self.current_voxel_map.shape[0])
            y_start = max(positions[id][1] - window_size[1] // 2, 0)
            y_end = min(positions[id][1] + window_size[1] // 2, self.current_voxel_map.shape[1])
            
            submap[id] = np.pad(
                self.current_voxel_map[x_start:x_end, y_start:y_end, :].astype(np.uint8)*255,
                ((max(0, -positions[id][0] + window_size[0] // 2), max(0, positions[id][0] + window_size[0] // 2 - self.current_voxel_map.shape[0])),
                 (max(0, -positions[id][1] + window_size[1] // 2), max(0, positions[id][1] + window_size[1] // 2 - self.current_voxel_map.shape[1])),
                 (0, 0)),
                mode='constant', constant_values=64
            )

        return submap
            

    def save_current_img(self):
        if self.img_idx % 10 == 0:
            fig, ax = visualize_voxel_map(
                self.current_voxel_map, show=False, drones=self.drones
            )
            elev = 60
            azim = 0
            ax.view_init(elev=elev, azim=azim)
            ax.dist = 7
            
            file_path = os.path.join(self.img_path, f'{self.img_idx // 10}.png')
            fig.savefig(file_path)
            
            # Explicitly clear and close the figure
            plt.close('all')

        self.img_idx += 1
        
    def _obs(self, is_first=False, is_last=False, is_terminal=False):
        views = self.get_drones_view(self._window_size) 

        image = None
        for i in range(self._num_drones):
            if image is None:
                image = views[i]
            else:
                image = np.concatenate((image, views[i]), axis=2)

        return dict(
            image = image,
            position = self.normalise_position(self.get_positions()).flatten(),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
    def normalise_position(self,position):
        normalised_position = np.zeros_like(position, dtype=np.float32)
        for i in range(3):
            normalised_position[:,i] = position[:,i] / self.gt_voxel_map.shape[i]
        return normalised_position

    def action_to_waypoint(self, action):
        waypoints = np.array_split(action, len(self.drones))
        int_waypoints = []
        for waypoint,drone in zip(waypoints,self.drones):
            if np.any(waypoint > 1.0) or np.any(waypoint < 0.0):
                raise ValueError('Waypoint values must be between 0.0 and 1.0')
            for i in range(3):
                # waypoint[i] = max(min(int(waypoint[i] * self.gt_voxel_map.shape[i]), self.gt_voxel_map.shape[i]-1),0)
                waypoint[i] = max(min(int(waypoint[i] * (self.max_steps*2+1) - self.max_steps) + drone.location[i], self.gt_voxel_map.shape[i]-1),0)
            int_waypoints.append(waypoint)
            waypoints = np.array(int_waypoints, dtype=int)
        return waypoints
    
    def get_positions(self):
        return np.array([drone.location for drone in self.drones]).reshape(-1, 3)

    def move_all_drones(self, NNwaypoints):
        """
        Moves all drones one by one. Each drone takes one step at a time, and the voxel map is updated after each step.
        """
        def any_steps_left(steps_per_drone):
            for steps in steps_per_drone:
                if len(steps) == 0:
                    return False
            return True
        
        if len(self.drones) != len(NNwaypoints):
            raise ValueError("Number of drones and number of waypoints do not match")
        
        [drone.set_goal(goal) for drone, goal in zip(self.drones, NNwaypoints)]

        steps_per_drone = [[] for _ in self.drones]
        for i, drone in enumerate(self.drones):
            steps = a_star_3d(self.current_voxel_map, drone.location,NNwaypoints[i])
            if steps is not None:
                steps_per_drone[i] = steps
            else:
                # raise ValueError(f"BFS failed for drone {i}")
                steps_per_drone[i] = []

        #local function to check if any of the drones have any steps left
        total_steps_performed = np.zeros(len(self.drones))

        # if self.log_images:
        #     self.save_current_img()
        while any_steps_left(steps_per_drone):
            for drone_index, drone in enumerate(self.drones):
                step = np.array(steps_per_drone[drone_index])[0]
                steps_per_drone[drone_index] = steps_per_drone[drone_index][1:]
                if not drone.perform_step(np.array(step)):#logic: if drone cannot move to the next step, replan and loose possibility to move in this loop, because while it replans, other drone are moving.
                    hold = a_star_3d(self.current_voxel_map, drone.location, NNwaypoints[drone_index])
                    steps_per_drone[drone_index] = hold
                else:
                    total_steps_performed[drone_index] += 1
            if self.log_images:
                self.save_current_img()
        return total_steps_performed


class Drone:# Drone(self.gt_voxel_map,self.gt_voxel_map_pointer, self.current_voxel_map, self.exploration_map, np.array(pos), id, self.lidar_rays_pointer,self.num_rays, self.lidar_range)
    def __init__(self, gt_voxel_map,gt_voxel_pointer,current_voxel_map, exploration_map, location, id, lidar_range, lidar_rays_pointer,num_rays):
        """
        Initializes the drone with a given voxel map and starting position.
        :param current_voxel_map: 3D numpy array representing the voxel space (0: free, 1: obstacle).
        :param start_position: Tuple (x, y, z) representing the starting position of the drone.
        """
        self.gt_voxel_map = gt_voxel_map
        self.map_size = gt_voxel_map.shape
        self.gt_voxel_pointer = gt_voxel_pointer
        self.current_voxel_map = current_voxel_map
        self.exploration_map = exploration_map
        self.lidar_range = lidar_range
        self.location = np.array(location)
        self.id = id
        # self.outer_shell_coords = outer_shell_coords
        self.lidar_rays_pointer = lidar_rays_pointer
        self.num_rays = num_rays
        self.goal = None
        self.path = [self.location]

    def set_goal(self, goal):
        """
        Set the goal for the drone to reach.
        :param goal: Tuple (x, y, z) representing the goal location.
        """
        self.goal = np.array(goal)
    
    def perform_step(self, next_position):
        """
        Convert the difference between current_position and next_position into a movement command.
        
        :param current_position: Tuple (x, y, z) representing current location.
        :param next_position: Tuple (x, y, z) representing the next location in the path.
        """

        if self.goal is None:
            raise ValueError("Goal is not set for the drone {self.id}")
        
        # Calculate the difference between current and next positions
        x_diff = next_position[0] - self.location[0]
        y_diff = next_position[1] - self.location[1]
        z_diff = next_position[2] - self.location[2]
        # print(f"Drone {self.id} tries to go to: {next_position}, delta x={x_diff}, delta y={y_diff} delfta z={z_diff}")  
        if abs(x_diff) <=1 and abs(y_diff)<=1 and abs(z_diff)<=1:
            if self.is_legal_move(next_position):
                execute_lidar_rays(self.gt_voxel_pointer, self.map_size[0],self.map_size[1],self.map_size[2],self.exploration_map, self.location,self.lidar_range, self.lidar_rays_pointer, self.num_rays)
                # mock_lidar(self.exploration_map,self.location,self.lidar_range)
                self.location = next_position
                self.current_voxel_map[:,:,:] = self.exploration_map & self.gt_voxel_map

                self.path.append(next_position)
                return True
            else:
                return False#planning failed
        else:
            return False
        
    def reached_goal(self):
        return np.array_equal(self.location, self.goal)
    
    def is_legal_move(self, new_position):#TRUE/FALSE legal move or not
        x, y, z = new_position
        if (0 <= x < self.gt_voxel_map.shape[0] and 
            0 <= y < self.gt_voxel_map.shape[1] and 
            0 <= z < self.gt_voxel_map.shape[2]):
            return self.current_voxel_map[int(x), int(y), int(z)] == 0
        return False
    

def VolumeToSurface(volume, start_locations):
    surface_map = np.zeros_like(volume, dtype=bool)
    list_of_visited_points = []
    visited = np.zeros_like(volume, dtype=bool)
    for i in start_locations:
        x, y, z = i
        assert(volume[x, y, z] == 0)
        if not visited[x, y, z]:
            visited[x, y, z] = True
            for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nz < volume.shape[2]:
                    if volume[nx, ny, nz] == 1:
                        surface_map[nx, ny, nz] = True
                        visited[nx, ny, nz] = True
                    elif volume[nx, ny, nz] == 0:
                        visited[nx, ny, nz] = True
                        list_of_visited_points.append((nx, ny, nz))
        while list_of_visited_points:
            x, y, z = list_of_visited_points.pop()
            for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nz < volume.shape[2]:
                    if volume[nx, ny, nz] == 1:
                        surface_map[nx, ny, nz] = True
                    elif not visited[nx, ny, nz]:
                        visited[nx, ny, nz] = True
                        list_of_visited_points.append((nx, ny, nz))

    return surface_map

if __name__ == "__main__":
    #ROUGH TRAINING LOOP
    map_size = (64, 64, 2)
    voxel_map = np.zeros((map_size[0], map_size[1], map_size[2]), dtype=bool)
    voxel_map[:,:,0] = 1
    spawn_points = np.argwhere(voxel_map == 0)
    start1 = spawn_points[0,:]
    print(type(start1))
    start2 = spawn_points[-1,:]
    NNwaypoints = (start1, start2)
    NNwaypoints = np.array(NNwaypoints)
    #initialize drones
    # print(start1, start2, NNwaypoints)

    Agents = MoveDrones(voxel_map, [start1, start2], lidar_range=40)
    submap = MoveDrones.get_drones_view(Agents)
    #print(submap)
    visualize_voxel_maps(Agents.current_voxel_map, Agents.gt_surface_map)
    #move drones
    NNwaypoints = Agents.move_all_drones((start2, start1))
    visualize_voxel_maps(Agents.current_voxel_map, Agents.gt_surface_map)