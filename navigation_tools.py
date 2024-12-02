import numpy as np
from collections import deque
from visualisations import visualize_voxel_map
import time
from heapq import heappush, heappop

def a_star_3d(voxel_map, start_pos, goal_pos):#GOAL is part of the steps, START is not
    """
    Performs A* search on a 3D voxel map.

    :param voxel_map: 3D numpy array (boolean) representing the voxel space (False: empty, True: obstacle).
    :param start_pos: numpy array (x, y, z) representing the starting position.
    :param goal_pos: numpy array (x, y, z) representing the goal position.
    :return: List of numpy arrays representing the shortest path from start to goal,
             or to the closest position if the goal is unreachable.
    """
    # Ensure start and goal are tuples for indexing
    start_pos = tuple(start_pos)
    goal_pos = tuple(goal_pos)
    
    dims = voxel_map.shape
    
    # Check if start or goal is invalid
    assert(not voxel_map[start_pos]), "Current drone position is an obstacle"
    if voxel_map[start_pos] or np.array_equal(start_pos, goal_pos):
        return []  # No path if start or goal is same as start
    
    # Define directions for movement (6-connected 3D space)
    directions = np.array([(1, 0, 0), (-1, 0, 0),
                           (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)])
    
    # Priority queue for A* (min-heap), initialized with the start position
    open_set = []
    heappush(open_set, (0, start_pos))  # (f_score, position)
    
    # Dictionaries for tracking scores and paths
    g_score = {start_pos: 0}  # Cost from start to current position
    f_score = {start_pos: np.linalg.norm(np.array(start_pos) - np.array(goal_pos))}  # Heuristic cost
    came_from = {}
    
    # Visited set to avoid re-expanding nodes
    visited = set()

    # Perform A* search
    while open_set:
        # Get the node with the lowest f_score
        _, current = heappop(open_set)
        
        # If we've reached the goal, reconstruct the path
        if current == goal_pos:
            path = []
            while current in came_from:
                path.append(np.array(current))
                current = came_from[current]
            #path.append(np.array(start_pos)) #uncomment if you want current pos to be the first path element, now returns path from start to goal without start
            return path[::-1]
        
        visited.add(current)
        
        # Explore neighbors
        for direction in directions:
            neighbor = tuple(np.array(current) + direction)
            
            # Check if the neighbor is within bounds and not an obstacle
            if (0 <= neighbor[0] < dims[0] and
                0 <= neighbor[1] < dims[1] and
                0 <= neighbor[2] < dims[2] and
                not voxel_map[neighbor]):
                
                # Skip already visited nodes
                if neighbor in visited:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + 1  # Uniform cost for each step
                
                # If this path is better, record it
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal_pos))
                    
                    # Add neighbor to the priority queue
                    heappush(open_set, (f_score[neighbor], neighbor))
    
    # If we exit the loop without finding the goal, return empty list
    return []

def bfs_3d(voxel_map, start_pos, goal_pos):#OLD STUPID CODE
    """
    Performs Breadth-First Search (BFS) on a 3D voxel map.
    
    :param voxel_map: 3D numpy array (boolean) representing the voxel space (False: empty, True: obstacle).
    :param start_pos: numpy array representing the starting position.
    :param goal_pos: numpy array representing the goal position.
    :return: List of numpy arrays representing the shortest path from start to goal, or to the closest position if goal is unreachable.
    """
    # Ensure start and goal are tuples for indexing purposes
    start_pos = tuple(start_pos)
    goal_pos = tuple(goal_pos)
    
    # Dimensions of the voxel map
    dims = voxel_map.shape
    
    # Check if the start is an obstacle
    if voxel_map[start_pos] or np.array_equal(start_pos, goal_pos):
        return []  # No path if the start is blocked or same as goal
    # Directions for 6 possible movements in 3D (up, down, left, right, forward, backward)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    # Queue for BFS (stores current position and path to that position)
    queue = deque([(start_pos, [start_pos])])
    
    # Set to keep track of visited positions
    visited = set([start_pos])

    # Variable to store the closest position to the goal
    closest_position = start_pos
    closest_distance = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))

    # BFS Loop
    while queue:
        current_pos, path = queue.popleft()
        
        # Check the neighbors of the current position
        for direction in directions:
            neighbor = (current_pos[0] + direction[0],
                        current_pos[1] + direction[1],
                        current_pos[2] + direction[2])
            
            # Ensure the neighbor is within bounds
            if 0 <= neighbor[0] < dims[0] and 0 <= neighbor[1] < dims[1] and 0 <= neighbor[2] < dims[2]:
                # Check if the neighbor is the goal
                if neighbor == goal_pos:
                    if voxel_map[neighbor]:  # If the goal is an obstacle
                        # print("I suspect NN chose a coordinate which it knew was an obstacle, so punish it!")
                        return []  # No path to the goal
                    else:
                        return path + [goal_pos]  # Return the path to the goal
                
                # Check if the neighbor is a free voxel and not visited
                if not voxel_map[neighbor] and neighbor not in visited:
                    visited.add(neighbor)  # Mark neighbor as visited
                    queue.append((neighbor, path + [neighbor]))  # Add neighbor to queue with updated path

                    # Update the closest position if this neighbor is closer to the goal
                    dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(neighbor))
                    if dist_to_goal < closest_distance:
                        closest_distance = dist_to_goal
                        closest_position = neighbor

    # If we exit the loop without finding the goal, return the path to the closest position
    # print("Goal unreachable, returning the path to the closest position")
    return []#bfs_3d_reconstruct_path(start_pos, closest_position, visited)



#--------RRT STAR FUNCTIONS----------------
# Steer towards the sampled point (step size fixed to 1) with axis-aligned movement
def steer(from_point, to_point):
    if np.array_equal(from_point, to_point):
        return from_point  # Already at the target

    # Calculate the direction vector
    direction = to_point - from_point
    
    # Choose the direction with the largest component to make a step
    step_direction = np.zeros(3, dtype=int)  # Initialize a zero vector
    if abs(direction[0]) >= abs(direction[1]) and abs(direction[0]) >= abs(direction[2]):
        step_direction[0] = np.sign(direction[0])  # Move along x-axis
    elif abs(direction[1]) >= abs(direction[0]) and abs(direction[1]) >= abs(direction[2]):
        step_direction[1] = np.sign(direction[1])  # Move along y-axis
    else:
        step_direction[2] = np.sign(direction[2])  # Move along z-axis

    # Move one step in the chosen direction
    new_point = from_point + step_direction
    return new_point


# Check if a point is free in the voxel space
def is_free(voxel_map, point):
    x, y, z = point
    return voxel_map[x, y, z] == 0  # CHANGED: checks if the point is free


# RRT* algorithm
def rrt_star(voxel_map, start, goal, max_iters=3000, goal_sample_rate=0.25):
    nodes = [(np.array(start), 0, None)]  # Start node: (point, cost, parent_index)
    goal_node = None
    
    for i in range(max_iters):  # CHANGED: loop continues until max_iters is reached
        if i%100 ==0:   print(i)
        if np.random.random() < goal_sample_rate:
            rand_point = np.array(goal)
        else:
            rand_point = np.random.randint(0, voxel_map.shape[0], size=3)  # CHANGED: random sample within map bounds

        # Find nearest node
        node_coords = np.array([node[0] for node in nodes])
        distances = np.linalg.norm(node_coords - rand_point, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_node = nodes[nearest_idx]

        # Steer towards the sampled point with a step of size 1
        new_point = steer(nearest_node[0], rand_point)
        if is_free(voxel_map, new_point):  # CHANGED: checks if the new point is free before adding
            new_cost = nearest_node[1] + 1  # Step cost is 1
            new_node = (new_point, new_cost, nearest_idx)
            nodes.append(new_node)

            # Rewiring logic for optimization
            for idx, node in enumerate(nodes):
                if not np.array_equal(node[0], new_node[0]) and np.linalg.norm(node[0] - new_point) <= 1:
                    potential_new_cost = new_cost + np.linalg.norm(new_point - node[0])
                    if potential_new_cost < node[1]:
                        nodes[idx] = (node[0], potential_new_cost, len(nodes) - 1)

            # Check if the new point is close to the goal
            if np.array_equal(new_point, goal):
                if not is_free(voxel_map, goal):  # CHANGED: check if goal is occupied
                    print(f"Goal is obstructed, assuming drone reached goal but staying at {nearest_node[0]}")  # CHANGED
                    goal_node = nearest_node  # CHANGED: stay at the last free point if goal is obstructed
                else:
                    goal_node = new_node
                break
        elif np.array_equal(goal, new_point) and not is_free(voxel_map, new_point):
            print("Goal is seen but is enreachable. Staying in current position... and?")
            goal_node = nearest_node
        # CHANGED: If goal is not reached within 10,000 iterations
        if i == max_iters - 1:
            print("Goal unreachable within the maximum iterations.")  # CHANGED: print unreachable goal message
            break  # CHANGED: exit the loop if goal isn't reachable

    # Reconstruct the path from goal to start
    if goal_node is not None:
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node[0])
            current_node = nodes[current_node[2]] if current_node[2] is not None else None
        path = path[::-1]  # Reverse path to go from start to goal
        return np.array(path[1:])  # Exclude the starting point
    else:
        print("No path found")  # CHANGED: updated message if no path is found
        return None  # No path found
    

if __name__ == "__main__":
    # Create a 3D voxel map (10x10x10) with some obstacles
    voxel_map = np.zeros((10, 10, 10), dtype=bool)
    voxel_map[1:2, 0:9, 0:10] = True  # Add an obstacle
    voxel_map[5:6, 1:10, 0:10] = True  # Add an obstacle
    voxel_map[7:10, 8:9, 0:10] = True  # Add an obstacle
    #visualize_voxel_map(voxel_map)
    # Define start and goal points
    start_pos = np.array([0, 0, 0])
    goal_pos = np.array([9, 9, 9])



    # Run the RRT* algorithm
    Time = time.time()
    path = a_star_3d(voxel_map, start_pos, goal_pos)
    print(path)
    print("Time taken: ", time.time()-Time)
    if path is not None:
        for point in path:
            voxel_map[tuple(point)] = True
        visualize_voxel_map(voxel_map)
    else:
        print("No path found")