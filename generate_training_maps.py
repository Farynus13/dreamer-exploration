import argparse
from newworldbuilding import schematic_to_boolean_array
from raycasting import visualize_3d_environment

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
        #voxel_map = np.zeros((args.map_size*args.map_scale, args.map_size*args.map_scale, 2), dtype=bool)
        path = os.path.dirname(os.path.abspath(__file__))

        """ 3D """
        # folder: /home/sapience/.minecraft/config/worldedit/schematics   old
        schematic_file = os.path.join(path, "schematics", "1.schematic")
        print("schematic_file: ", schematic_file)
        voxel_map = schematic_to_boolean_array(schematic_file)
        # Resize voxel_map to 64x64x27 by padding with zeros
        padded_voxel_map = np.zeros((64, 64, 27), dtype=bool)
        padded_voxel_map[:voxel_map.shape[0], :voxel_map.shape[1], :voxel_map.shape[2]] = voxel_map
        voxel_map = padded_voxel_map
        voxel_map[:,:,0] = 0

        print("yo")
        visualize_3d_environment(voxel_map)
        #np.save(os.path.join(args.output_dir, f'map_{i}.npy'), voxel_map)
        