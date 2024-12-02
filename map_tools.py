from noise import pnoise2
import numpy as np
import random
import matplotlib.pyplot as plt

def get_letter_matrix(letter=[]):
    """
    Generates a random letter from the alphabet as a 2D boolean numpy array.
    
    :param size: The size of the output array (must be a multiple of 3).
    :return: 2D boolean numpy array with 1 representing the ink and 0 representing the free space.
    """
    
    # Predefined 3x3 templates for letters
    letter_templates = {
        'A': np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [1, 0, 1]], dtype=bool),
        
        'B': np.array([[1, 1, 0],
                       [1, 1, 1],
                       [1, 1, 0],
                       [1, 1, 1],
                       [1, 1, 0]], dtype=bool),
        
        'C': np.array([[1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 1, 1]], dtype=bool),
        
        'D': np.array([[1, 1, 0],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 0]], dtype=bool),
        
        'E': np.array([[1, 1, 1],
                       [1, 0, 0],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 1, 1]], dtype=bool),
        
        'F': np.array([[1, 1, 1],
                       [1, 0, 0],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0]], dtype=bool),
        
        'G': np.array([[0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1]], dtype=bool),
        
        'H': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [1, 0, 1]], dtype=bool),
        
        'I': np.array([[1, 1, 1],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [1, 1, 1]], dtype=bool),
        
        'J': np.array([[0, 1, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1]], dtype=bool),
        
        'K': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 0, 1],
                       [1, 0, 1]], dtype=bool),
        
        'L': np.array([[1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 1, 1]], dtype=bool),
        
        'M': np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype=bool),
        
        'N': np.array([[1, 0, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 0, 1]], dtype=bool),
        
        'O': np.array([[0, 1, 0],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool),
        
        'P': np.array([[1, 1, 0],
                       [1, 1, 1],
                       [1, 1, 0],
                       [1, 0, 0],
                       [1, 0, 0]], dtype=bool),
        
        'Q': np.array([[0, 1, 0],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 0],
                       [0, 1, 1]], dtype=bool),
        
        'R': np.array([[1, 1, 0],
                       [1, 1, 1],
                       [1, 1, 0],
                       [1, 0, 1],
                       [1, 0, 1]], dtype=bool),
        
        'S': np.array([[0, 1, 1],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0]], dtype=bool),
        
        'T': np.array([[1, 1, 1],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]], dtype=bool),
        
        'U': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=bool),
        
        'V': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool),
        
        'W': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1]], dtype=bool),
        
        'X': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1],
                       [1, 0, 1]], dtype=bool),
        
        'Y': np.array([[1, 0, 1],
                       [1, 0, 1],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]], dtype=bool),
        
        'Z': np.array([[1, 1, 1],
                       [0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 1]], dtype=bool),
    }
    if letter:
        return letter_templates[letter]
    # Choose a random letter
    letter = random.choice(list(letter_templates.keys()))
    
    # Get the corresponding 3x3 template
    template = letter_templates[letter]

    return template


def combine_letters(x=64,y=64,z=1):
    # Define the target array size (64x64)
    letter_width = 3
    letter_height = 5
    spacing = 1  # Space between letters

    # Define the words with their respective letters
    sapience = ['O', 'M', 'M', 'S', 'A', 'P', 'I', 'E', 'N', 'C', 'E', 'M', 'M', 'M', 'O']
    space =    ['M', 'O', 'M', 'J', 'M', 'J', 'M', 'J', 'M', 'J', 'M', 'F', 'M', 'S', 'M']
    milosz =   ['M', 'N', 'I', 'L', 'O', 'S', 'Z', 'M', 'G', 'G', 'G', 'M', 'G', 'G', 'M']
    jason =    ['M', 'J', 'A', 'S', 'O', 'N', 'M', 'J', 'S', 'B', 'M', 'F', 'M', 'X', 'M']
    jakub =    ['M', 'J', 'A', 'K', 'U', 'B', 'M', 'M', 'X', 'X', 'X', 'B', 'M', 'X', 'M']
    luc_thomas=['M', 'L', 'U', 'C', 'M', 'T', 'H', 'O', 'N', 'A', 'S', 'M', 'B', 'X', 'M']
    ewoud =    ['M', 'E', 'W', 'O', 'U', 'D', 'M', 'M', 'R', 'R', 'R', 'B', 'M', 'X', 'M']
    angelo =   ['M', 'A', 'N', 'G', 'E', 'L', 'O', 'M', 'R', 'R', 'R', 'B', 'M', 'X', 'M']
    leonardo = ['M', 'L', 'E', 'O', 'N', 'A', 'R', 'D', 'O', 'M', 'R', 'B', 'M', 'X', 'M']
    simon =    ['M', 'S', 'I', 'N', 'O', 'N', 'M', 'R', 'R', 'R', 'R', 'B', 'M', 'X', 'M']

    # Combine all the words to form the list of words
    combined = [space, sapience, milosz, jason, jakub, luc_thomas, ewoud, angelo, leonardo, simon]

    # Create an empty canvas of size 64x64
    canvas = np.zeros((x, y), dtype=bool)

    # Set the starting position for each word and row
    row_spacing = letter_height + spacing

    # Iterate over each word (each row)
    for row_index, letters in enumerate(combined):
        row_start = row_index * row_spacing  # Calculate the starting row

        # If the word goes beyond the canvas height, stop (prevent overflow)
        if row_start + letter_height > y:
            break

        # Iterate over each letter in the word
        col_start = 0
        for letter in letters:
            # Fetch the letter matrix
            letter_matrix = get_letter_matrix(letter)

            # If the next letter exceeds the canvas width, break (prevent overflow)
            if col_start + letter_width > x:
                break

            # Place the letter in the appropriate position in the canvas
            canvas[row_start:row_start + letter_height, col_start:col_start + letter_width] = letter_matrix

            # Move to the next column, adding space between letters
            col_start += letter_width + spacing

    # Convert to a 3D voxel map (for visualization purposes)
    canvas_3d = np.repeat(canvas[:, :, np.newaxis], z, axis=2)  # Duplicate across 3rd dimension for RGB visualization
    #combine(canvas_3d)
    return canvas_3d

def combine(voxel_map):
    xsize, ysize, _ = voxel_map.shape
    for i in range(xsize//4):
        for j in range(ysize//4):
            if random.random() < 0.4:
                kernel = get_letter_matrix('M')
            else:
                kernel = get_letter_matrix()
            kernel = np.repeat(kernel[:, :, np.newaxis], 1, axis=2)
            top_left_y = np.random.randint(0, ysize - kernel.shape[1] + 1)
            top_left_x = np.random.randint(0, xsize - kernel.shape[0] + 1)
            #print(kernel.shape[0], top_left_x, top_left_y)
            voxel_map[top_left_x:top_left_x + kernel.shape[0], top_left_y:top_left_y + kernel.shape[1]] = kernel
    return voxel_map

def perlin_noise_2Dmask(shape, scale=10, threshold=0.04, seed=0):#3D with z=1
    """
    Generate a 2D boolean mask based on Perlin noise.
    
    Parameters:
        shape (tuple): Shape of the output array (height, width).
        scale (float): Scale factor for the Perlin noise; larger values make features larger.
        threshold (float): Threshold for selecting areas; values above the threshold will be `True`.
        seed (int): Seed for the Perlin noise for reproducibility.
    
    Returns:
        numpy.ndarray: A boolean mask with `True` for selected areas.
    """
    #np.random.seed(seed)
    noise_seed_x, noise_seed_y = np.random.randint(0, 100, 2)
    height, width, z = shape
    if z != 1:
        raise ValueError("Only 2D masks are supported.")
    mask = np.zeros((height, width), dtype=bool)
    mask = np.repeat(mask[:, :, np.newaxis], 1, axis=2)
    # Generate Perlin noise and apply the threshold
    for y in range(height):
        for x in range(width):
            # Generate a Perlin noise value at (x, y) coordinates, scaled
            noise_val = pnoise2(x / scale + noise_seed_x, y / scale + noise_seed_y, octaves=4)
            # Apply threshold to determine selected areas
            mask[y, x, 0] = noise_val > threshold
    
    return mask

def generate_maze(width, height,z=1):
    # Ensure even dimensions by adding 1 to odd dimensions for walls.
    if width % 2 == 1:
        width += 1
    if height % 2 == 1:
        height += 1

    # Initialize maze with walls (1's)
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Function to check if a cell is within bounds and not visited
    def is_valid(x, y):
        return 0 <= x < width and 0 <= y < height and maze[y][x] == 1

    # Directions for moving in the maze: (dx, dy)
    directions = [(0, 2), (2, 0), (-2, 0), (0, -2)]

    def carve(x, y):
        stack = [(x, y)]  # Start the stack with the initial position
        maze[y][x] = 0  # Mark the initial cell as empty

        while stack:
            cx, cy = stack[-1]  # Current cell (last element of the stack)
            random.shuffle(directions)  # Randomize direction order for variation

            carved = False  # Flag to track if a new path was carved
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny):
                    # Carve a path between the current cell and the new cell
                    maze[cy + dy // 2][cx + dx // 2] = 0
                    maze[ny][nx] = 0  # Mark the new cell as empty
                    stack.append((nx, ny))  # Add the new cell to the stack
                    carved = True
                    break  # Move to the new cell immediately

            if not carved:
                stack.pop()  # Backtrack if no valid moves are possible


    # Start the carving process from a random starting point inside the maze bounds
    start_x, start_y = (random.randrange(1, width-1, 2), random.randrange(1, height-1, 2))
    carve(start_x, start_y)

    maze = np.array(maze)

    # extend to 3 dims  
    maze_3d = np.zeros((width, height, z), dtype=bool)
    for i in range(z):
        maze_3d[:,:,i] = maze
    return maze_3d