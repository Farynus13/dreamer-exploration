
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk

def create_gif_from_images(directory, output_path, duration,delete_img=False):
    """
    Create a GIF from all images in a directory.
    
    Parameters:
        directory (str): Path to the directory containing images.
        output_path (str): Path to save the output GIF.
        duration (int): Duration for each frame in milliseconds.
    """
    # Get all image file paths
    image_files = [
        os.path.join(directory, file)
        for file in sorted(os.listdir(directory))
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    # Sort them according to numbering not lexicographically
    image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # Ensure there are images in the directory
    if not image_files or len(image_files)< 2:
        print(f"{len(image_files)} images found in the directory. Gif requires at least 2")
        return

    # here implement the code to create a gif from the images
    images = [Image.open(file) for file in image_files]
    
    images[0].save(
        output_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0
                )

    if delete_img:
        for file in image_files:
            os.remove(file)

    print(f"GIF saved to {output_path}")

def visualize_3d_array(arr: np.ndarray, title: str = None, show: bool = True, drones=None, off_screen: bool = False):
    x,y,z, = arr.shape
    arr[0,0,0]=1
    arr[x-1,y-1,z-1]=1
    """
    Visualize a 3D boolean NumPy array with enhanced visibility and navigation:
    - Color gradient along z-axis for height distinction
    - Optimized initial isometric camera view
    - Scalar bar for height reference
    - Orientation axes
    - Subtle edges with reduced thickness
    - Partial transparency for depth perception
    - First-person navigation using arrow keys and mouse

    Parameters:
    - arr (np.ndarray): A 3D boolean NumPy array (1s = solid, 0s = empty).
    - title (str, optional): Text to display on the plot (e.g., filename).
    - show (bool, optional): Whether to display the plot.
    - drones (list, optional): List of drone objects with paths to visualize.
    - off_screen (bool, optional): Whether to render off-screen.

    Raises:
    - ValueError: If the input is not a 3D boolean array.

    Navigation Controls:
    - Arrow keys: Move forward (up), backward (down), left, right
    - Mouse: Click and drag to look around
    """
    # Validate input
    if arr.ndim != 3 or arr.dtype != bool:
        raise ValueError("Input must be a 3D boolean NumPy array")
 
    # Pad the array to ensure edges are captured
    padded_arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)

    # Create a PyVista ImageData object
    grid = pv.ImageData()
    grid.dimensions = padded_arr.shape
    grid.origin = (-0.5, -0.5, -0.5)  # Center voxels
    grid.spacing = (1, 1, 1)          # Unit voxel size

    # Assign values as point data
    grid.point_data["values"] = padded_arr.astype(float).ravel(order="F")

    # Extract isosurface
    mesh = grid.contour([0.5])

    # Add height scalars for coloring
    mesh.point_data['height'] = mesh.points[:, 2]

    # Create plotter
    plotter = pv.Plotter(window_size=[800, 600], off_screen=off_screen)

    # Add mesh with visibility enhancements
    plotter.add_mesh(
        mesh,
        scalars='height',
        cmap='viridis',
        show_edges=True,
        edge_color='lightgray',
        line_width=0.3,
        opacity=0.85,
        ambient=0.2,
        diffuse=0.8,
        specular=0.3,
        specular_power=15
    )

    # Add scalar bar
    plotter.add_scalar_bar(title='Height (z-axis)')

    # Set initial isometric camera view
    bounds = (0, arr.shape[0], 0, arr.shape[1], 0, arr.shape[2])
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    center = ((x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2)
    distance = max(x_max - x_min, y_max - y_min, z_max - z_min) * 1.5
    camera_position = (center[0] + distance, center[1] + distance, center[2] + distance)
    plotter.camera_position = [camera_position, center, (0, 0, 1)]

    # Add orientation axes
    plotter.add_axes()

    # Add coordinate system axes
    plotter.show_axes()

    # Add title if provided
    if title:
        plotter.add_text(title, position='upper_edge', font_size=12)

    # Add navigation instructions
    plotter.add_text(
        "Navigation: Arrow keys to move, mouse to look around",
        position='lower_left',
        font_size=10
    )

    # Set interactor style to flight
    flight_style = vtk.vtkInteractorStyleTrackballCamera()
    plotter.interactor_style = flight_style

    if show:
        plotter.show()
    else:
        return plotter

def visualize_voxel_map(voxel_map, show=True, drones=None):
    """
    Visualize a 3D voxel map with optional drone paths.
    
    Parameters:
    - voxel_map: 3D numpy array (boolean), indicating the presence of voxels.
    - show: bool, whether to display the plot.
    - drones: list of drone objects, each with a `path` attribute (list of 3D coordinates).
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects.
    """
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract dimensions of the voxel map
    x_size, y_size, z_size = voxel_map.shape
    max_distance = np.sqrt(x_size**2 + y_size**2 + z_size**2)
    
    # Generate the grid for the entire voxel map
    x, y, z = np.indices((x_size + 1, y_size + 1, z_size + 1))
    
    # Mask the grid for active voxels
    active_voxels = voxel_map > 0
    
    # Compute colors only for active voxels
    red = y[:-1, :-1, :-1][active_voxels] / y_size
    green = z[:-1, :-1, :-1][active_voxels] / z_size
    blue = x[:-1, :-1, :-1][active_voxels] / x_size
    
    distances = np.sqrt(
        x[:-1, :-1, :-1][active_voxels]**2 +
        y[:-1, :-1, :-1][active_voxels]**2 +
        z[:-1, :-1, :-1][active_voxels]**2
    )
    opacity = 0.1 + 0.9 * (distances / max_distance)
    
    colors = np.zeros((*voxel_map.shape, 4), dtype=float)
    colors[active_voxels] = np.stack([red, green, blue, opacity], axis=-1)
    
    # Plot voxels
    ax.voxels(
        voxel_map, facecolors=colors,
        edgecolor=(0.7, 0.7, 0.7, 0.15), linewidth=0.5
    )
    
    # Plot drone paths if provided
    if drones:
        drone_colors = ['r', 'b', 'c', 'm']
        for i, drone in enumerate(drones):
            color = drone_colors[i % len(drone_colors)]
            path = drone.path if len(drone.path) < 300 else drone.path[-300:]
            path = np.array(path)
            
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2)
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=color, s=100)
            ax.scatter(drone.goal[0], drone.goal[1], drone.goal[2], color=color, s=10, marker='x')
            
    # Set labels and adjust aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([x_size, y_size, z_size])
    ax.set_xlim(-5, x_size + 5 )
    ax.set_ylim(-5, y_size + 5 )
    ax.set_zlim(-5, z_size + 5 )
    
    # Show the plot if needed
    if show:
        plt.show()
    
    return fig, ax


def helper():
    base_path = os.path.join(os.path.dirname(__file__), 'maps3d2')
    _maps = os.listdir(maps_path)
    map_idx = np.random.randint(len(os.listdir(maps_path)))
    voxel_map = np.load(os.path.join(maps_path, _maps[map_idx], _maps[map_idx]))

if __name__ == "__main__":
    # Example usage: replace with your folder path
    folder_path = "maps3d2/"
    file_paths = os.listdir(folder_path)
    # Load and visualize the 594_port.npy file
    for file_path in file_paths:
        if file_path.endswith(".npy"):
            if file_path == "1246_netherforteca2.npy":
                file_path = os.path.join(folder_path, file_path, 'starting_locations.npy')
                print(f"Visualizing {file_path}...")
                voxel_map = np.load(file_path)
                visualize_3d_array(voxel_map)
    # filename = "138_???.npy"
    # file_path = os.path.join(folder_path, filename , filename)
    # print(f"Visualizing {filename}...")
    # voxel_map = np.load(file_path)
    # print(voxel_map.shape)
    # visualize_3d_array(voxel_map)