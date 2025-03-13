import numpy as np
from main import MoveDrones
from visualisations import create_gif_from_images
import os
from our_dreamer.dreamer_model import get_model
import argparse
from visualisations import visualize_3d_array


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Example usage:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Drone exploration simulation.')
    #parser.add_argument('--map_size', type=int, nargs=1, default=(64, 64, 24), help='Size of the map (x, y, z)')
    parser.add_argument('--window-size', type=int, nargs=1, default=(64, 64, 24), help='Size of the map (x, y, z)')
    parser.add_argument('--num_drones', type=int, default=2, help='Number of drones')
    parser.add_argument('--lidar_num_h_rays', type=int, default=36, help='Number of horizontal lidar rays')
    parser.add_argument('--lidar_num_v_rays', type=int, default=36, help='Number of vertical lidar rays')
    parser.add_argument('--lidar_fov_v', type=int, default=180, help='Vertical field of view of the lidar')
    parser.add_argument('--lidar_range', type=int, default=12, help='Range of the lidar')
    parser.add_argument('--model_path', type=str, default='logdir/drone3dv3/latest.pt', help='Path to the model file')

    args = parser.parse_args()
    if os.path.exists('gifs'):
        for file in os.listdir('gifs'):
            file_path = os.path.join('gifs', file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        print(f"Deleted contents of gifs directory")
    mapdir = 'maps3d2/'
    map = '28_cos.npy'
    voxel_map = np.load(os.path.join(mapdir, map, map))
    surface_map = np.load(os.path.join(mapdir, map ,'surface_map.npy'))
    map_size = voxel_map.shape
    num_drones = args.num_drones
    lidar_params = {
        "num_h_rays": args.lidar_num_h_rays,
        "num_v_rays": args.lidar_num_v_rays,
        "fov_v": args.lidar_fov_v
    }
    lidar_range = args.lidar_range
    model_path = args.model_path
    window_size = tuple(args.window_size)
    spawn_points = np.argwhere(voxel_map == 0)
    np.random.shuffle(spawn_points)
    spawn_points = spawn_points[:num_drones]

    Agents = MoveDrones(gt_voxel_map=voxel_map, gt_surface_map=surface_map,
                        start_positions=spawn_points,
                        lidar_range=lidar_range,
                        windows_size=window_size,
                        lidar_params=lidar_params,
                        num_drones=num_drones,
                        log_images=True,
                        img_path='gifs/')
        
    model = get_model(size=(64,64),z=3,num_drones=num_drones,path=model_path)

    agent_state = None # no initial state
    step = 0
    gt_surface_sum = np.sum(Agents.gt_surface_map)
    while np.sum(Agents.current_voxel_map) <= 0.95 * gt_surface_sum and step < 75:
        print(f"Step {step}  |  Exploration: {np.sum(Agents.current_voxel_map) / gt_surface_sum * 100:.2f}%")
        step += 1
        obs = [Agents._obs()]
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = model(obs, agent_state)

        action = np.array(action['action'][0].detach().cpu())        
        # natively the action was normalized to [-1,1], we need to scale it back to [0,1]
        action = (action + 1) / 2

        waypoints = Agents.action_to_waypoint(action)
        print(f"Waypoints: {waypoints}")
        # else:
        #     break
            # waypoints = deterministic_benchmark(Agents.exploration_map, Agents.get_positions())
            # waypoints = efficient_frontier_exploration(Agents.exploration_map, Agents.get_positions())

        # print(f"Waypoints: {waypoints}")
        Agents.move_all_drones(waypoints)


    #create gif from images and delete images
    

    create_gif_from_images('gifs', 'gifs/dreamer.gif', 50, True)
    # os.system('rm gifs/*.png')

