import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training maps for the drone environment')
    parser.add_argument('--num_maps', type=int, default=1, help='Number of maps to generate')
    parser.add_argument('--map_size', type=int, default=64, help='Size of the map')
    parser.add_argument('--map_scale', type=int, default=1, help='Scale of the map')
    parser.add_argument('--output_dir', type=str, default='maps', help='Output directory for the maps')
    args = parser.parse_args()

    import os
    import numpy as np
    from map_tools import generate_maze,perlin_noise_2Dmask


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        #clear the directory
        for f in os.listdir(args.output_dir):
            os.remove(os.path.join(args.output_dir, f))

    for i in range(args.num_maps):
        # voxel_map = generate_maze(args.map_size*args.map_scale, args.map_size*args.map_scale, 1)
        # mask = perlin_noise_2Dmask([args.map_size*args.map_scale, args.map_size*args.map_scale,1],np.random.randint(8, 13),np.random.uniform(-0.02,0.02))#only works for z=1 for n 
        # voxel_map = np.where(mask,voxel_map,0)
        #generate voxel map with only floor
        voxel_map = np.zeros((args.map_size*args.map_scale, args.map_size*args.map_scale, 2), dtype=bool)
        voxel_map[:,:,0] = 1

        np.save(os.path.join(args.output_dir, f'map_{i}.npy'), voxel_map)