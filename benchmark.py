import numpy as np
import matplotlib.pyplot as plt
from main import MoveDrones,generate_maze
from visualisations import create_gif_from_images
import os
from our_dreamer.dreamer_model import get_model
from map_tools import perlin_noise_2Dmask
import argparse
from tqdm import tqdm
# Example usage:

import numpy as np

def deterministic_benchmark(exploration_map, drone_positions,lidar_range):
    """
    Determine the next waypoint for each drone for exploration.
    
    Parameters:
        voxel_map (np.ndarray): 3D binary voxel map (0: unexplored, 1: explored).
        drone_positions (list): List of (x, y, z) tuples for drone positions.
        lidar_range (int): Range of the lidar sensor.
    
    Returns:
        list: List of (x, y, z) waypoints for each drone.
    """
    available_voxels = np.argwhere(exploration_map == 0)  # Indices of unexplored voxels

    waypoints = []


    for i,pos in enumerate(drone_positions):
        if len(available_voxels) == 0:
            waypoints.append(tuple(pos))
        else:
            distances = np.linalg.norm(available_voxels - np.array(pos), axis=1)
            closest_voxel = available_voxels[np.argmin(distances)]

            # Remove everything within the lidar range from the available voxels so that the drones don't collide
            available_voxels = available_voxels[np.linalg.norm(available_voxels - np.array(pos), axis=1) > lidar_range]
            waypoints.append(tuple(closest_voxel))

    return waypoints

def execute_agent_benchmark(Agents, model, max_steps, exploration_percentage):
    """
    Execute the agent benchmark for exploration.
    
    Parameters:
        agents (MoveDrones): Instance of MoveDrones class.
        model_path (str): Path to the trained model.
    """        

    agent_state = None # no initial state
    obs = [Agents._obs(is_first=True)]
    step = 0
    agent_distance = 0
    surface_map_sum = np.sum(Agents.gt_surface_map)
    while np.sum(Agents.current_voxel_map) <= exploration_percentage * surface_map_sum and step < max_steps:
        # print(f"Step {step}  |  Exploration: {np.sum(Agents.current_voxel_map) / np.sum(Agents.gt_surface_map) * 100:.2f}%")
        step += 1
        if np.sum(Agents.current_voxel_map) < 0.99 * surface_map_sum:
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
            action, agent_state = model(obs, agent_state)

            action = np.array(action['action'][0].detach().cpu())        
            # natively the action was normalized to [-1,1], we need to scale it back to [0,1]
            action = (action + 1) / 2

            waypoints = Agents.action_to_waypoint(action)

            # print(f"Waypoints: {waypoints}")
        else:
            waypoints = deterministic_benchmark(Agents.exploration_map, Agents.get_positions(), Agents.lidar_range)
            # waypoints = efficient_frontier_exploration(Agents.exploration_map, Agents.get_positions())

        total_steps_performed = Agents.move_all_drones(waypoints)
        obs = [Agents._obs()]

        agent_distance += np.sum(total_steps_performed)



    return agent_distance,np.sum(Agents.current_voxel_map) / surface_map_sum, step

def execute_deterministic_benchmark(Agents, max_steps, exploration_percentage):
    """
    Execute the deterministic benchmark for exploration.
    
    Parameters:
        agents (MoveDrones): Instance of MoveDrones class.
    """
    step = 0
    agent_distance = 0
    surface_map_sum = np.sum(Agents.gt_surface_map)
    while np.sum(Agents.current_voxel_map) <= exploration_percentage * surface_map_sum and step < max_steps:
        # print(f"Step {step}  |  Exploration: {np.sum(Agents.current_voxel_map) / np.sum(Agents.gt_surface_map) * 100:.2f}%")
        step += 1
        waypoints = deterministic_benchmark(Agents.exploration_map, Agents.get_positions(), Agents.lidar_range)
        # waypoints = efficient_frontier_exploration(Agents.exploration_map, Agents.get_positions())

        total_steps_performed = Agents.move_all_drones(waypoints)
        agent_distance += np.sum(total_steps_performed)

    return agent_distance,np.sum(Agents.current_voxel_map) / surface_map_sum, step
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Benchmark drone exploration.')
    parser.add_argument('--map_size', type=int, nargs=3, default=(64, 64, 2), help='Size of the map (x, y, z)')
    parser.add_argument('--num_drones', type=int, default=3, help='Number of drones')
    parser.add_argument('--lidar_num_h_rays', type=int, default=32, help='Number of horizontal lidar rays')
    parser.add_argument('--lidar_num_v_rays', type=int, default=32, help='Number of vertical lidar rays')
    parser.add_argument('--lidar_fov_v', type=int, default=180, help='Vertical field of view of the lidar')
    parser.add_argument('--lidar_range', type=int, default=10, help='Range of the lidar')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs for the benchmarks')
    parser.add_argument('--model_path', type=str, default='logdir/drone-testing-script/latest.pt', help='Path to the trained model')
    parser.add_argument('--max_steps', type=int, default=120, help='Maximum number of steps for the benchmarks')
    parser.add_argument('--exploration_percentage', type=float, default=0.99, help='Exploration percentage threshold')
    args = parser.parse_args()

    model_path = args.model_path
    num_runs = args.num_runs
    map_size = tuple(args.map_size)
    num_drones = args.num_drones
    lidar_params = {
        "num_h_rays": args.lidar_num_h_rays,
        "num_v_rays": args.lidar_num_v_rays,
        "fov_v": args.lidar_fov_v
    }
    lidar_range = args.lidar_range
    max_steps = args.max_steps
    exploration_percentage = args.exploration_percentage

    # voxel_map = generate_maze(map_size[0], map_size[1], map_size[2])
    voxel_map = np.zeros((map_size[0], map_size[1], map_size[2]), dtype=bool)
    voxel_map[:,:,0] = 1

    #Degenerate maze new function
    # perlin_mask = perlin_noise_2Dmask((map_size[0], map_size[1], 1),np.random.randint(8, 13),np.random.uniform(-0.02,0.02))
    # voxel_map = np.where(perlin_mask, voxel_map, 0)
    available_spawn_points = np.argwhere(voxel_map == 0)
    deterministic_distances = []
    agent_distances = []

    deterministic_explo = []
    agent_explo = []

    deterministic_steps = []
    agent_steps = []

    model = get_model(size=(64,64),z=map_size[2],num_drones=num_drones,path=model_path)

    for _ in tqdm(range(num_runs), desc="Benchmark Runs",total=num_runs):
        np.random.shuffle(available_spawn_points)
        spawn_points = available_spawn_points[:num_drones]

        Agents = MoveDrones(gt_voxel_map=voxel_map,
                            start_positions=spawn_points,
                            lidar_range=lidar_range,
                            window_size=[64,64],
                            lidar_params=lidar_params,
                            num_drones=num_drones,
                            log_images=False)
        
        # Execute deterministic benchmark
        print("Executing deterministic benchmark...")
        deterministic_distance,deterministic_explo_percentage,deterministic_step = execute_deterministic_benchmark(Agents, max_steps, exploration_percentage)
        deterministic_distances.append(deterministic_distance)
        deterministic_explo.append(deterministic_explo_percentage)
        deterministic_steps.append(deterministic_step)
        print(f"Deterministic Exploration: {deterministic_explo_percentage * 100:.2f}%")

        Agents = MoveDrones(gt_voxel_map=voxel_map,
                        start_positions=spawn_points,
                        lidar_range=lidar_range,
                        window_size=[64,64],
                        lidar_params=lidar_params,
                        num_drones=num_drones,
                        log_images=False)
        
        # Execute agent benchmark
        print("Executing agent benchmark...")
        agent_distance,agent_explo_percentage,agent_step = execute_agent_benchmark(Agents, model, max_steps, exploration_percentage)
        agent_distances.append(agent_distance)
        agent_explo.append(agent_explo_percentage)
        agent_steps.append(agent_step)
        print(f"Agent Exploration: {agent_explo_percentage * 100:.2f}%")

    avg_deterministic_distance = np.mean(deterministic_distances)
    avg_agent_distance = np.mean(agent_distances)

    avg_deterministic_explo = np.mean(deterministic_explo)
    avg_agent_explo = np.mean(agent_explo)

    avg_deterministic_steps = np.mean(deterministic_steps)
    avg_agent_steps = np.mean(agent_steps)

    #create nice table
    print(f"{'Benchmark':<20}{'Deterministic':<20}{'Agent':<20}")
    print(f"{'Distance':<20}{avg_deterministic_distance:<20}{avg_agent_distance:<20}")
    print(f"{'Exploration':<20}{avg_deterministic_explo:<20}{avg_agent_explo:<20}")
    print(f"{'Steps':<20}{avg_deterministic_steps:<20}{avg_agent_steps:<20}")
    
