import numpy as np
import matplotlib.pyplot as plt
from main import MoveDrones,generate_maze
from visualisations import create_gif_from_images
import os
from our_dreamer.dreamer_model import get_model
from map_tools import perlin_noise_2Dmask
from benchmark import deterministic_benchmark
import argparse

# Example usage:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Drone exploration simulation.')
    parser.add_argument('--map_size', type=int, nargs=3, default=(64, 64, 2), help='Size of the map (x, y, z)')
    parser.add_argument('--num_drones', type=int, default=3, help='Number of drones')
    parser.add_argument('--lidar_num_h_rays', type=int, default=32, help='Number of horizontal lidar rays')
    parser.add_argument('--lidar_num_v_rays', type=int, default=32, help='Number of vertical lidar rays')
    parser.add_argument('--lidar_fov_v', type=int, default=180, help='Vertical field of view of the lidar')
    parser.add_argument('--lidar_range', type=int, default=10, help='Range of the lidar')
    parser.add_argument('--model_path', type=str, default='logdir/drone-testing-script/latest.pt', help='Path to the model file')

    args = parser.parse_args()

    map_size = tuple(args.map_size)
    num_drones = args.num_drones
    lidar_params = {
        "num_h_rays": args.lidar_num_h_rays,
        "num_v_rays": args.lidar_num_v_rays,
        "fov_v": args.lidar_fov_v
    }
    lidar_range = args.lidar_range
    model_path = args.model_path
    # voxel_map = generate_maze(map_size[0], map_size[1], map_size[2])
    voxel_map = np.zeros((map_size[0], map_size[1], map_size[2]), dtype=bool)
    voxel_map[:,:,0] = 1
    spawn_points = np.argwhere(voxel_map == 0)
    np.random.shuffle(spawn_points)
    spawn_points = spawn_points[:num_drones]
    print(spawn_points)
    #Degenerate maze new function
    # perlin_mask = perlin_noise_2Dmask((map_size[0], map_size[1], 1),np.random.randint(8, 13),np.random.uniform(-0.02,0.02))
    # voxel_map = np.where(perlin_mask, voxel_map, 0)


    Agents = MoveDrones(gt_voxel_map=voxel_map,
                        start_positions=spawn_points,
                        lidar_range=lidar_range,
                        window_size=[64,64],
                        lidar_params=lidar_params,
                        num_drones=num_drones,
                        log_images=True,
                        img_path='gifs/')
        
    model = get_model(size=(64,64),z=map_size[2],num_drones=num_drones,path=model_path)

    agent_state = None # no initial state
    step = 0
    gt_surface_sum = np.sum(Agents.gt_surface_map)
    while np.sum(Agents.current_voxel_map) <= 0.99 * gt_surface_sum and step < 120:
        print(f"Step {step}  |  Exploration: {np.sum(Agents.current_voxel_map) / gt_surface_sum * 100:.2f}%")
        step += 1
        if np.sum(Agents.current_voxel_map) < 0.9 * gt_surface_sum:
            obs = [Agents._obs()]
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
            action, agent_state = model(obs, agent_state)

            action = np.array(action['action'][0].detach().cpu())        
            # natively the action was normalized to [-1,1], we need to scale it back to [0,1]
            action = (action + 1) / 2

            waypoints = Agents.action_to_waypoint(action)
        else:
            waypoints = deterministic_benchmark(Agents.exploration_map, Agents.get_positions(), Agents.lidar_range)
            # waypoints = efficient_frontier_exploration(Agents.exploration_map, Agents.get_positions())
        # print(f"Waypoints: {waypoints}")
        Agents.move_all_drones(waypoints)
        

    #create gif from images and delete images


    create_gif_from_images('gifs', 'gifs/dreamer.gif', 100)
    os.system('rm gifs/*.png')

