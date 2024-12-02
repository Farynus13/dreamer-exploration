import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import ctypes
# Load the shared library
lib = ctypes.CDLL('./libbresenham3d.so')

# Define the Point3D structure in Python
class Point3D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("z", ctypes.c_int)]
# Define the argument and return types of the Bresenham3D function
#  int x_ini, int y_ini, int z_ini,
#   int lidar_range, 
#   int *num_points,
#   uint8_t *voxel_map, int submap_x, int submap_y, int submap_z,
#   Point3D *rays, int num_rays
lib.Bresenham3D.argtypes = [
    ctypes.c_int,  # x1
    ctypes.c_int,  # y1
    ctypes.c_int,  # z1
    ctypes.POINTER(ctypes.c_int),  # Pointer to int (num_points)
    ctypes.POINTER(ctypes.c_uint8),  # Pointer to flattened uint8 voxel map
    ctypes.c_int,  # submap_x
    ctypes.c_int,  # submap_y
    ctypes.c_int,   # submap_z
    ctypes.POINTER(ctypes.c_int32),  # Pointer to Point3D (rays)
    ctypes.c_int,  # num_rays
]
lib.Bresenham3D.restype = ctypes.POINTER(Point3D)  # Returns a pointer to Point3D

# Define the argument type for the FreePoints function
lib.FreePoints.argtypes = [ctypes.POINTER(Point3D)]

##########################################################################################################
def generate_lidar_rays(num_h_rays=124, num_v_rays=96,fov_v=80, lidar_range=20):
    fov_v = np.radians(fov_v)
    # Generate horizontal angles
    h_angles = np.linspace(0, 2*np.pi, num_h_rays, endpoint=True)
    # Generate vertical angles
    v_angles = np.linspace(-fov_v/2, fov_v/2, num_v_rays) 
    # return list of all produced rays
    h_angles = np.repeat(h_angles, num_v_rays)
    v_angles = np.tile(v_angles, num_h_rays)
    
    z = np.sin(v_angles)
    x = np.cos(v_angles) * np.cos(h_angles)
    y = np.cos(v_angles) * np.sin(h_angles)
    
    rays = np.vstack((x, y, z)).T * lidar_range
    return rays.astype(np.int32)

def execute_lidar_rays(gt_voxel_map_pointer, voxel_map_x, voxel_map_y, voxel_map_z, exploration_map, current_pos, lidar_range, rays_pointer, num_rays):
    #e(gt_voxel_map_lidar_execution_pointer, voxel_map_x, voxel_map_y, voxel_map_z, exploration_map, current_pos, lidar_range, lidar_rays_pointer, num_rays)
    x1, y1, z1 = current_pos

    num_points = ctypes.c_int((num_rays+6)*lidar_range)  # Create an integer to store the number of points
    #  int x_ini, int y_ini, int z_ini,
    #int lidar_range, 
    #int *num_points,
    #uint8_t *voxel_map, int submap_x, int submap_y, int submap_z,
    #Point3D *rays, int num_rays
    points_ptr = lib.Bresenham3D(x1, y1, z1, ctypes.byref(num_points), gt_voxel_map_pointer, voxel_map_x, voxel_map_y, voxel_map_z, rays_pointer, num_rays)
    if not points_ptr:
        raise MemoryError("Failed to allocate memory for points.")
    for i in range(num_points.value):
        exploration_map[points_ptr[i].x, points_ptr[i].y, points_ptr[i].z] = True
    lib.FreePoints(points_ptr)
    exploration_map[x1, y1, z1] = True  
    return exploration_map
#############################################################################################################
#below are functions that supported development of the above functions
def visualize_lidar_rays(rays):
    # Extract x, y, z coordinates from the rays
    x_vals = rays[:, 0]
    y_vals = rays[:, 1]
    z_vals = rays[:, 2]
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot rays as points in 3D space
    ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=20)
    
    # Set labels and plot details
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Visualization of LiDAR Rays')
    
    # Set aspect ratio to be equal for all axes
    ax.set_box_aspect([1, 1, 1])
    
    plt.show()

def visualize_voxel_maps(voxel_map, exploration_map):
    # Ensure both maps have the same shape
    
    # Set up the figure with two subplots side by side
    fig = plt.figure(figsize=(16, 7))
    
    # Extract dimensions of the voxel map (both maps have the same size)
    x_size, y_size, z_size = voxel_map.shape
    
    # Find the maximum distance for opacity normalization
    max_distance = np.sqrt(x_size**2 + y_size**2 + z_size**2)

    def get_voxel_colors(voxel_map):
        # Create normalized coordinate grids
        x, y, z = np.meshgrid(
            np.arange(x_size) / x_size,
            np.arange(y_size) / y_size,
            np.arange(z_size) / z_size,
            indexing='ij'
        )
        distance = np.sqrt(x**2 + y**2 + z**2) / max_distance

        # RGBA color channels
        red = y
        green = z
        blue = x
        opacity = 0.9 - 0.8 * distance

        # Combine channels into a single array
        colors = np.zeros(voxel_map.shape + (4,))
        colors[..., 0] = red
        colors[..., 1] = green
        colors[..., 2] = blue
        colors[..., 3] = opacity

        # Apply voxel_map mask to set fully transparent for non-voxels
        colors[~voxel_map] = [0, 0, 0, 0]
        return colors

    # Get the voxel colors for both maps
    colors1 = get_voxel_colors(voxel_map)
    colors2 = get_voxel_colors(exploration_map)

    # Plot the first voxel map on the left
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.voxels(voxel_map, facecolors=colors1, edgecolor=(0.7, 0.7, 0.7, 0.3), linewidth=0.5)
    ax1.set_title('Voxel Map')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(0, x_size)
    ax1.set_ylim(0, y_size)
    ax1.set_zlim(0, z_size)
    ax1.set_box_aspect([x_size, y_size, z_size])  # Aspect ratio matches the shape

    # Plot the second voxel map (exploration map) on the right
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.voxels(exploration_map, facecolors=colors2, edgecolor=(0.7, 0.7, 0.7, 0.3), linewidth=0.5)
    ax2.set_title('Exploration Map')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(0, x_size)
    ax2.set_ylim(0, y_size)
    ax2.set_zlim(0, z_size)
    ax2.set_box_aspect([x_size, y_size, z_size])  # Aspect ratio matches the shape

    # Function to synchronize rotation between the two plots
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
        fig.canvas.draw_idle()

    # Connect the mouse movement to the sync function
    fig.canvas.mpl_connect('motion_notify_event', on_move)

    # Show the plot with both voxel maps side by side
    plt.tight_layout()
    plt.show()




def visualize_3d_environment(voxel_map):
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract dimensions of the voxel map
    x_size, y_size, z_size = voxel_map.shape

    # Find the maximum distance for opacity normalization
    max_distance = np.sqrt(x_size**2 + y_size**2 + z_size**2)
    
    # Prepare an array to store colors
    colors = np.empty(voxel_map.shape + (4,), dtype=float)  # 4 for RGBA

    #print("drawing: ", voxel_map)
    
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if voxel_map[x, y, z]:
                    # Calculate color based on position
                    red = y / y_size  # Red increases with y
                    green = z / z_size  # Green increases with z
                    blue = x / x_size  # Blue increases with x
                    
                    # Calculate the opacity based on the distance from (0, 0, 0)
                    distance = np.sqrt(x**2 + y**2 + z**2)
                    opacity = 0.1 + 0.9 * (distance / max_distance)  # From 10% to 100%
                    
                    # Set the RGBA color (red, green, blue, alpha)
                    colors[x, y, z] = [red, green, blue, opacity]
                else:
                    # Set fully transparent for non-voxel cells
                    colors[x, y, z] = [0, 0, 0, 0]
    
    # Plot the voxels using the voxel_map boolean array and color array
    # Change the edgecolor to a light gray and set linewidth to make the edges thinner
    ax.voxels(voxel_map, facecolors=colors, edgecolor=(0.7, 0.7, 0.7, 0.3), linewidth=0.5)
    
    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Adjust the plot to make sure it's easy to view for larger environments
    #ax.set_box_aspect([x_size, y_size, z_size])  # Aspect ratio matches the shape If you want to display only what is in the map and not display empty 'edges' of the map
    #or fix the display size to the map size:
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_zlim(0, z_size)
    # Show the plot
    plt.show()

def visualize_voxel_map(environment, highlight_points=None, base_point_size=20):
    """
    Visualize a 3D boolean numpy array efficiently as a wireframe with dynamic point scaling.
    Adjusts opacity based on the distance from the camera to improve visibility of features.
    
    Parameters:
    - environment (numpy.ndarray): A 3D boolean numpy array where True indicates an occupied space.
    - base_point_size (int): Base size of points, which will scale with window size.
    - highlight_points (list of numpy.ndarray): List of arrays, where each array is a set of coordinates
      to highlight. Each array should be of shape (N, 3), where N is the number of special points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of all occupied (True) points
    occupied_indices = np.argwhere(environment)
    
    # For larger 3D environments, reduce the number of points plotted to improve speed
    step = max(1, len(occupied_indices) // 5000)  # Adjust step size based on density
    sparse_indices = occupied_indices[::step]
    
    # Extract x, y, z coordinates for the sparse set
    x, y, z = sparse_indices[:, 0], sparse_indices[:, 1], sparse_indices[:, 2]

    # Normalize coordinates to range [0, 255] for RGB mapping
    x_max, y_max, z_max = environment.shape
    colors = np.zeros((len(x), 3))
    colors[:, 0] = (x / x_max) * 255  # Red channel from X
    colors[:, 1] = (y / y_max) * 255  # Green channel from Y
    colors[:, 2] = (z / z_max) * 255  # Blue channel from Z
    colors = colors / 255  # Scale to [0, 1] for matplotlib

    # Initial scatter plot with base point size and color
    scatter = ax.scatter(x, y, z, s=base_point_size, c=colors, alpha=0.8)

    # Highlight special points
    if highlight_points:
        for points in highlight_points:
            hx, hy, hz = points[:, 0], points[:, 1], points[:, 2]
            ax.scatter(hx, hy, hz, s=base_point_size * 3, c='black', alpha=1.0, label="Special Points")

    # Function to update point size and opacity based on distance from the camera
    def update_point_size_and_opacity(event=None):
        # Calculate new point size based on the current figure width
        current_fig_width = fig.get_size_inches()[0] * fig.dpi
        new_point_size = base_point_size * (current_fig_width / 800)  # Adjust 800 as a baseline width
        scatter.set_sizes([new_point_size])  # Update scatter point size

        # Get the current camera position based on the projection
        proj = ax.get_proj()
        cam_pos = proj[:, 3][:3]  # Approximate camera position as the last column of the matrix

        # Calculate distance of each point to the camera position
        distances = np.sqrt((x - cam_pos[0])**2 + (y - cam_pos[1])**2 + (z - cam_pos[2])**2)
        
        # Normalize distances and invert for alpha (closer points are more opaque)
        max_distance = np.max(distances)
        alphas = 1 - (distances / max_distance)  # Scale to range [0, 1] for alpha
        
        # Set new alpha values for the scatter points
        scatter.set_alpha(None)  # Disable constant alpha for the collection
        scatter._facecolors[:, 3] = alphas  # Set individual alpha per point in face colors
        
        # Update the size of highlight points if they exist
        if highlight_points:
            for collection in ax.collections[1:]:  # Assumes the first collection is the main points
                collection.set_sizes([new_point_size * 3])

        fig.canvas.draw_idle()  # Redraw the canvas to apply the update

    # Connect the resize event to dynamically adjust point size and opacity
    fig.canvas.mpl_connect('resize_event', update_point_size_and_opacity)
    
    # Call the update function once to set the initial size and opacity
    update_point_size_and_opacity()

    # Set axis limits and aspect ratio for an accurate structure
    ax.set_box_aspect([environment.shape[0], environment.shape[1], environment.shape[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.legend()
    plt.show()

# Create an exemplary 3D voxel environment (20x20x10) with some shapes
def create_example_environment(x,y,z):
    voxel_map = np.zeros((x,y,z), dtype=bool)
    # Add a shape
    voxel_map[0:x-1, 1:3, 0:z-1] = True

    # Add a shape
    voxel_map[x//2:x//2+1, 0:y, 0:z] = True
    voxel_map[x-1:x, 0:y//2+1, z//2:z//2] = True
    voxel_map[x//2-3:x//2-2, 0:y, 0:z] = True
    voxel_map[0:x//2-2, y//2:y//2+1, 1:z] = True
    

    # Add a diagonal plane
    #for i in range(8):
    #    voxel_map[5+i, 5+i, i] = True
    
    return voxel_map

def mock_lidar(exploration_map,current_pos,lidar_range):
    x1, y1, z1 = current_pos
    x_range = np.arange(max(0, x1 - lidar_range), min(exploration_map.shape[0], x1 + lidar_range))
    y_range = np.arange(max(0, y1 - lidar_range), min(exploration_map.shape[1], y1 + lidar_range))
    z_range = np.arange(max(0, z1 - lidar_range), min(exploration_map.shape[2], z1 + lidar_range))

    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    distances = np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)
    mask = distances <= lidar_range

    exploration_map[x[mask], y[mask], z[mask]] = True
    return exploration_map

# Define vectors
def unit_vector_test():
    vectors = generate_lidar_rays(num_h_rays=96, num_v_rays=16, fov_v=20)
    origin = np.ones((vectors.shape[0], 3)) * 1.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(vectors.shape[0]):
        ax.quiver(
            origin[i,0], origin[i,1], origin[i,2],
            vectors[i,0], vectors[i,1], vectors[i,2],
            arrow_length_ratio=0.1
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    plt.show()

#testing
if __name__ == "__main__":

    voxel_map = create_example_environment(45,45,45)
    exploration_map1 = np.full_like(voxel_map, False, dtype=bool)
    exploration_map2 = np.full_like(voxel_map, False, dtype=bool)
    next_pos = (29,28,29)
    current_pos = (29, 29, 29)
    x1, y1, z1 = current_pos
    x2, y2, z2 = next_pos
    print(current_pos)

    num_h_rays = 256
    num_v_rays = 256
    fov_v = 90
    lidar_range = 10
    voxel_map[:, :, 28] = True

    start_time = time.time()
    lidar_rays = generate_lidar_rays(num_h_rays, num_v_rays, fov_v, lidar_range)
    num_rays = len(lidar_rays)#lidar params h*v
    print(f"Number of rays: {num_rays} should equal {num_h_rays*num_v_rays}")
    lidar_rays_pointer = lidar_rays.flatten().astype(ctypes.c_int32)
    lidar_rays_pointer = lidar_rays_pointer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    voxel_map_x, voxel_map_y, voxel_map_z = voxel_map.shape
    gt_voxel_map_lidar_execution_pointer = voxel_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    assert gt_voxel_map_lidar_execution_pointer is not None
    assert lidar_rays_pointer is not None
    exploration_map = execute_lidar_rays(gt_voxel_map_lidar_execution_pointer, voxel_map_x, voxel_map_y, voxel_map_z, exploration_map1, current_pos, lidar_range, lidar_rays_pointer, num_rays)

    exploration_map = mock_lidar(exploration_map2,current_pos,lidar_range)

    print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

    voxel_map[current_pos]=True
    exploration_map[current_pos]=True
    visualize_voxel_maps(exploration_map1, exploration_map2)
    #exploration_map = np.logical_not(exploration_map) #only here is the exploration map inverted of easier visualization
    #visualize_voxel_maps(voxel_map, exploration_map)
    #visualize_3d_environment(voxel_map)
    #drone_locations = [np.array([[15, 15, 15], [0, 5, 10], [25, 25, 25]])]
    #visualize_voxel_map(voxel_map, drone_locations)