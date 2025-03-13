from main import VolumeToSurface
from visualisations import visualize_3d_array, visualize_voxel_map
import os
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def find_pockets(original):
    """
    Identify connected pockets of free space in a 3D boolean NumPy array and return their subarrays.
    
    Parameters:
    - original (np.ndarray): 3D boolean array where False (0) is free space and True (1) is an obstacle.
    
    Returns:
    - list: List of 3D boolean subarrays, each representing a pocket of connected free spaces
            and their adjacent obstacles.
    """
    # Get the shape of the input array
    shape = original.shape
    
    # Initialize a visited array to track explored free spaces
    visited = np.zeros(shape, dtype=bool)
    
    # List to store the subarrays for each pocket
    pockets = []
    
    # Define the 6 directions for 3D connectivity (up, down, left, right, forward, backward)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    # Iterate over all positions in the 3D array
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # Start a new pocket if we find an unvisited free space
                if not original[i, j, k] and not visited[i, j, k]:
                    # Initialize the subarray for this pocket
                    subarray = np.zeros(shape, dtype=bool)
                    
                    # Use a queue for BFS to explore the connected free spaces
                    queue = deque([(i, j, k)])
                    visited[i, j, k] = True
                    subarray[i, j, k] = True
                    
                    # Perform BFS to find all connected free spaces and adjacent walls
                    while queue:
                        ci, cj, ck = queue.popleft()  # Current position
                        
                        # Check all 6 neighboring positions
                        for di, dj, dk in directions:
                            ni, nj, nk = ci + di, cj + dj, ck + dk
                            
                            # Ensure the neighbor is within bounds
                            if (0 <= ni < shape[0] and 
                                0 <= nj < shape[1] and 
                                0 <= nk < shape[2]):
                                if not original[ni, nj, nk] and not visited[ni, nj, nk]:
                                    # If neighbor is free and unvisited, add to queue and subarray
                                    queue.append((ni, nj, nk))
                                    visited[ni, nj, nk] = True
                                    subarray[ni, nj, nk] = True
                                elif original[ni, nj, nk]:
                                    # If neighbor is a wall, add to subarray but not queue
                                    subarray[ni, nj, nk] = True
                    
                    # Add the completed subarray to the list of pockets
                    pockets.append(subarray)
    return pockets


def prepdataset():
    base_path = os.path.join(os.path.dirname(__file__), 'maps3d2')
    for subdir in os.listdir(base_path):
        print(f"Processing directory: {subdir}")
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path) and subdir == "938_liniowiec.npy":
            npy_file = os.path.join(subdir_path, f'{subdir}')
            if os.path.isfile(npy_file):
                data = np.load(npy_file)
                backup_file = os.path.join(subdir_path, 'orginal.npy')
                os.rename(npy_file, backup_file)
                print(f"Processing file: {npy_file}")
                assert np.all(np.isin(data, [0, 1])), "Data contains values other than 0 and 1"
                pockets = find_pockets(data)
                for i, pocket in enumerate(pockets):
                    assert pocket.shape == data.shape, f"Pocket {i} has a different shape from the original data"
                if len(pockets) == 1:
                    pass
                else:
                    if sum(np.sum(pocket) > 0.25 * np.prod(data.shape) for pocket in pockets) >= 2:
                        print("Warning: There are at least 2 pockets with a size greater than 30% of the original array. {npy_file}")
                    if max(np.sum(pocket) for pocket in pockets) <= 0.5 * np.prod(data.shape):
                        
                        print("Warning: The dominant pocket is not bigger than 50% of the original array. {npy_file}")
                    # Remove the biggest pocket from the set
                    biggest_pocket = max(pockets, key=np.sum)
                    biggest_pocket_index = next(i for i, pocket in enumerate(pockets) if np.array_equal(pocket, biggest_pocket))
                    pockets.pop(biggest_pocket_index)
                    for pocket in pockets:
                        data[pocket] = 1
            starting_point = np.argwhere(data == 0)
            np.random.shuffle(starting_point)
            starting_point = starting_point[0]
            surface_map = VolumeToSurface(data, start_locations=[starting_point])
            output_file = os.path.join(subdir_path, 'surface_map.npy')
            #visualize_3d_array(surface_map)
            assert np.all(np.isin(surface_map, [0, 1])), "Surface map contains values other than 0 and 1"
            assert surface_map.shape == data.shape, "Surface map has a different shape from the original data"
            np.save(npy_file, data)
            print(f"Saving surface map to {output_file}")
            np.save(output_file, surface_map)

def check(base_path = '/home/sapience/workspaces/ExplorationMultiAgent/playground/maps3d2'):
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders in {base_path}")
    count = 0
    for subfolder in subfolders:
        files = os.listdir(subfolder)
        required_files = [f"{os.path.basename(subfolder)}", "orginal.npy", "surface_map.npy"]
        if all(file in files for file in required_files):
            print(f"Subfolder {subfolder} GOOD")
            count += 1
        else:
            print(f"Subfolder {subfolder} is missing some required files.")
    print(f"Found {count} out of {len(subfolders)} subfolders with all required files.")

def create_starting_locations():
    base_path = os.path.join(os.path.dirname(__file__), 'maps3d2')
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    for subfolder in subfolders:
        npy_file = os.path.join(subfolder, f"{os.path.basename(subfolder)}")
        if os.path.isfile(npy_file):
            data = np.load(npy_file)
            starting_locations = np.zeros_like(data)
            
            # Define the 6 directions for 3D connectivity (up, down, left, right, forward, backward)
            directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            
            # Iterate over all positions in the 3D array
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        if data[i, j, k] == 0:
                            for di, dj, dk in directions:
                                ni, nj, nk = i + di, j + dj, k + dk
                                if (0 <= ni < data.shape[0] and 
                                    0 <= nj < data.shape[1] and 
                                    0 <= nk < data.shape[2] and 
                                    data[ni, nj, nk] == 1):
                                    starting_locations[i, j, k] = 1
                                    break
            
            output_file = os.path.join(subfolder, 'starting_locations.npy')
            np.save(output_file, starting_locations)
            print(f"Saved starting locations to {output_file}")



if __name__ == '__main__':
    #prepdataset()
    create_starting_locations()
    #visualize_3d_array(arr = np.load('/home/sapience/workspaces/ExplorationMultiAgent/playground/maps3d2/938_liniowiec.npy/938_liniowiec.npy'))







