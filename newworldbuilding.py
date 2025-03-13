import numpy as np
import nbtlib
import os
from visualisations import visualize_3d_array, visualize_voxel_map
from create_dataset import find_pockets


def schematic_to_boolean_array(schematic_file):
    # Load the schematic file using nbtlib
    try:
        schematic = nbtlib.load(schematic_file)
    except KeyError as e:
        print(f"Error loading schematic file {schematic_file}: {e}")
        return None

    # Extract key tags: Width, Height, Length, and Blocks
    width = schematic['Width']
    height = schematic['Height']
    length = schematic['Length']
    blocks = schematic['Blocks']  # Block IDs (0 for air, non-0 for solid)
    #print(width, height, length, len(blocks))

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

def load_and_modify_array(array):
    stop = False
    while not stop:
        action = input("Do you want to delete a slice from top or bottom or stop or revert? (t/b/s/r): ").lower()
        if action not in ['t', 'b', 's', 'r']:
            print("Invalid input. Please enter 't' to delete from top, 'b' to delete from bottom, or 's' to stop.")
            continue
        if action == 's':
            stop = True
            continue
        if action == 't':
            slices_to_remove = int(input("Enter the number of z slices to remove from the top: "))
            if slices_to_remove >= array.shape[2]:
                print(f"Cannot remove {slices_to_remove} slices. The array only has {array.shape[2]} slices.")
                continue
            hold = array.copy()
            array = array[:, :, :-slices_to_remove]
            visualize_3d_array(array)
            print(f"Modified array shape: {array.shape}")
        if action == 'b':
            slices_to_remove = int(input("Enter the number of z slices to remove from the bottom: "))
            if slices_to_remove >= array.shape[2]:
                print(f"Cannot remove {slices_to_remove} slices. The array only has {array.shape[2]} slices.")
                continue
            hold = array.copy()
            array = array[:, :, slices_to_remove:]
            visualize_3d_array(array)
            print(f"Modified array shape: {array.shape}")
        if action == 'r':
            if not 'hold' in locals():
                print("Cannot revert. No previous state to revert to.")
                continue
            array = hold
    return array
def sortoutschematics():
    path = os.path.dirname(os.path.abspath(__file__))
    schematics_folder = os.path.join(path, "schematics")
    schematic_files = sorted([f for f in os.listdir(schematics_folder) if f.endswith(".schematic") and f.split('.')[0].isdigit()], key=lambda x: int(x.split('.')[0]))
    skipfirst = 670
    for filename in schematic_files:
        if int(filename.split('.')[0]) < skipfirst:
            continue
        schematic_file = os.path.join(schematics_folder, filename)
        array = schematic_to_boolean_array(schematic_file)
        if array is None:
            print(f"SRemoving {filename} due to loading error.")
            os.remove(schematic_file)
            continue
        #print(filename, ": ", array.shape)
        if array.shape[2] < 25 or array.shape[0] < 64 or array.shape[1] < 64 or array.shape[0]*array.shape[1] < 128*128:
            print(f"Skipping {filename} because its height is less than 25 or is too small")
            continue
        # Calculate the z coordinate (slice) with the highest number of non-zero elements
        new_filename = filename.replace('.schematic', 'v1.schematic')
        new_schematic_file = os.path.join(schematics_folder, new_filename)
        os.rename(schematic_file, new_schematic_file)
        schematic_file = new_schematic_file
        # z_counts = np.sum(array, axis=(0, 1))
        # print(z_counts, z_counts.shape)
        # max_z = np.argmax(z_counts)
        # print(f"The z coordinate with the highest number of non-zero elements is: {max_z}")
        visualize_3d_array(array)
        save = input(f"Do you want to save {filename} file y/n or edit e?: ")
        if save.lower() == 'n':
            comment = input("Enter a comment for this schematic: ")
            output_filename = f"{filename.split('.')[0]}_{comment}.schematic"
            output_path = os.path.join(path, "schematics", output_filename)
            os.rename(schematic_file, output_path)
            print(f"Skipping saving for {filename}.")
            continue
        if save.lower() == 'e':
                array = load_and_modify_array(array)
        comment = input("Enter a comment for this schematic: ")
        output_filename = f"{filename.split('.')[0]}_{comment}.npy"
        output_path = os.path.join(path, "maps3d", output_filename)
        np.save(output_path, array)

def prep_training_dataset():
    pass

if __name__ == "__main__":
    base_path = '/home/sapience/workspaces/ExplorationMultiAgent/playground/maps3d2'
    filename = '256_dungeons.npy'
    file_path = os.path.join(base_path, filename, filename)
    data = np.load(file_path)
    pockets = find_pockets(data)
    pockets = sorted(pockets, key=np.sum, reverse=True)
    for i, pocket in enumerate(pockets):
        pocket[0,0,0] = 1
        pocket[pocket.shape[0]-1, pocket.shape[1]-1, pocket.shape[2]-1] = 1
        visualize_3d_array(pocket)