import numpy as np
import random
from raycasting import visualize_voxel_maps, visualize_voxel_map
from scipy.ndimage import convolve, correlate

def generate_3d_maze(x_size, y_size, z_size):

    if 1==1:#random.random() < 0.5:

        #if x_size % 2 == 1 and y_size % 2 == 1 and z_size % 2 == 1:
        #       print("maze may not be good")
        maze = np.ones((x_size, y_size, z_size), dtype=bool)

        # Directions for moving in the 3D maze (x, y, z)
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        # Helper function to check if a position is within maze bounds
        def is_valid(x, y, z):
            return 0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size and maze[x][y][z] == True

        # Stack-based DFS algorithm
        def iterative_dfs(start_x, start_y, start_z):
            # Initialize a stack with the starting cell

            stack = [(start_x, start_y, start_z)]
            maze[start_x][start_y][start_z] = False  # Mark the starting cell as empty space
            while stack:
                # Pop the current cell from the stack
                x, y, z = stack.pop()

                #If this cell has already been visited, skip it
                #if maze[x, y, z] == False:
                #    continue
                # Mark the current cell as empty space

                random.shuffle(directions)  # Shuffle directions at each step

                for dx, dy, dz in directions:
                    nx, ny, nz = x + dx * 2, y + dy * 2, z + dz * 2  # Move 2 steps to ensure space between walls
                    if is_valid(nx, ny, nz):
                        # Add neighbor coordinates and the wall between them to the list
                        maze[nx][ny][nz] = False  # Remove the wall between the cells
                        maze[x + dx][y + dy][z + dz] = False  # Remove the wall between the cells
                        stack.append((nx, ny, nz))  # Add the neighbor to the stack

        # Start DFS from a random cell on the grid that is odd-numbered (so we have space for walls)
        start_x = random.randint(0, (x_size - 1) // 2) * 2
        start_y = random.randint(0, (y_size - 1) // 2) * 2
        start_z = random.randint(0, (z_size - 1) // 2) * 2
        
        # Begin DFS from the random start point
        iterative_dfs(start_x, start_y, start_z)      #iterative_dfs(start_x, start_y, start_z)
        
        return maze
    else:
        #different mazegen

        return maze

def degenerate_maze(maze):
        kernel = np.array([[[1]], 
                       [[1]],
                       [[1]],
                       [[1]],
                       [[1]],
                       [[1]],
                       [[1]],
                       [[1]],
                       [[1]]])
        #print(kernel.shape)
        maze = apply_kernel(maze, kernel)
        #visualize_voxel_map(maze)
        kernel = np.array([[[1], [1], [1], [1], [1], [1],[1], [1], [1]]])
        #print(kernel.shape)
        maze = apply_kernel(maze, kernel)
        return maze


def apply_kernel(maze, kernel):
    # Ensure maze and kernel are integer arrays
    test_int = maze.astype(int)
    kernel_int = kernel.astype(int)

    assert np.sum(kernel_int) == len(kernel_int.flatten()), "kernel can only contain 1s"

    # Adjust kernel to be 3D to match the dimensions of the maze
    #kernel_3d = kernel_int[:, :,np.newaxis]  # Expanding kernel to be 3D with shape (3, 1, 1)

    # Perform correlation to find where the kernel matches the test array
    result = correlate(test_int, kernel_int, cval=0)
    #print(f"{result=}")

    # Check where the result matches the sum of the kernel (i.e., exact match)
    match_locations = np.where(result == np.sum(kernel_int))

    #print(f"{match_locations=}")

    # For each matching location, flip the values in the test array
    for row, col, depth in zip(*match_locations):
        # Compensate for padding in correlate
        row -= kernel_int.shape[0] - 1
        col -= kernel_int.shape[1] - 1
        depth -= kernel_int.shape[2] - 1
        
        # Define the region based on kernel shape
        row_end = row + kernel_int.shape[0]
        col_end = col + kernel_int.shape[1]
        depth_end = depth + kernel_int.shape[2]

        # Log the selected region before the operation
        selected_region_before = maze[row:row_end, col:col_end, depth:depth_end]

        #print(f"Selected region from (row={row}:{row_end}, col={col}:{col_end}, depth={depth}:{depth_end}):")
        #print(f"Before operation:\n{selected_region_before}")

        # Perform the operation (setting values to 0)
        maze[row:row_end, col:col_end, depth:depth_end] = 0

        # Log the selected region after the operation
        selected_region_after = maze[row:row_end, col:col_end, depth:depth_end]

        #print(f"After operation:\n{selected_region_after}\n")
    return maze

if __name__ == "__main__":
    # Kernel and voxel map
    voxel_map = generate_3d_maze(10, 10, 10)
    visualize_voxel_map(voxel_map)
    voxel_map = degenerate_maze(voxel_map)

    #voxel_map = voxel_map[:, :, np.newaxis]  # Convert to 3D by adding an extra axis

    # Apply the kernel

    x=20
    y=20
    z=20
    #voxel_map = generate_3d_maze(x, y, z)
    #voxel_map = np.zeros((x,y,z), dtype=bool)
    #voxel_map[0:2,:,:] []= True
    #voxel_map = erode_maze(voxel_map)
    #voxel_map[0:5,10:19:,:] = 0

    visualize_voxel_map(voxel_map)#, voxel_map2)

    #visualize_voxel_map(voxel_map)
