import numpy as np
import nbtlib
from raycasting import visualize_3d_environment

def schematic_to_boolean_array(schematic_file):
    # Load the schematic file using nbtlib
    schematic = nbtlib.load(schematic_file)

    # Extract key tags: Width, Height, Length, and Blocks
    width = schematic['Width']
    height = schematic['Height']
    length = schematic['Length']
    blocks = schematic['Blocks']  # Block IDs (0 for air, non-0 for solid)
    print(width, height, length, len(blocks))

    # Create a 3D numpy array with shape (width, length, height) to set z as the vertical axis
    boolean_array = np.zeros((width, length, height), dtype=bool)

    # Fill the boolean array based on the block IDs
    index = 0
    for z in range(height):
        for y in range(length):
            for x in range(width):
                # In this orientation, z is the vertical axis, with (x, y) as the ground plane
                boolean_array[x, y, z] = blocks[index] != 0
                index += 1

    return boolean_array


# Example usage:
# folder: /home/sapience/.minecraft/config/worldedit/schematics
schematic_file = '/home/sapience/.minecraft/config/worldedit/schematics/test2.schem'
array = schematic_to_boolean_array(schematic_file)
print(array.shape)
#visualize_voxel_map(array)
visualize_3d_environment(array)
