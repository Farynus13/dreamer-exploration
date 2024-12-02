
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

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