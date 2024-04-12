import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import argparse
import math
import io
import imageio
from PIL import Image

# Get the particle's starting position.
@jit(nopython=True, cache=True)
def get_starting_position(N, structure_radius, x_middle, y_middle):
    x = -1
    y = -1
    while x >= N or x < 0 or y >= N or y < 0:
        # Use polar coordinates to generate a position around the edge of the DLA structure.
        theta = random.random() * 2 * math.pi
        x = int((1.2 * structure_radius) * math.cos(theta) + x_middle)
        y = int((1.2 * structure_radius) * math.sin(theta) + y_middle)
    return x, y

# Check whether the particle is near the structure in any direction.
@jit(nopython=True, cache=True)
def is_near_structure(x, y, matrix, N, M):
    if x - 1 >= 0 and matrix[x - 1][y] == M: return True
    if x + 1 < N and matrix[x + 1][y] == M: return True
    if y - 1 >= 0 and matrix[x][y - 1] == M: return True
    if y + 1 < N and matrix[x][y + 1] == M: return True
    return False

# Simulate the aggregation of N particles.
@jit(nopython=True, cache=True)
def simulate_particles(matrix, N, M, structure_radius, kill_proportion, p, n):
    i = 0
    x_middle = N / 2
    y_middle = N / 2
    while i < n:
        # Spawn a particle.
        x, y = get_starting_position(N, structure_radius, x_middle, y_middle)
        # Follow the particle until it touches the structure M times and attaches to it.
        while matrix[x][y] < M:
            # Get a random movement for the particle.
            old_x, old_y = x, y
            random_move = random.randint(0, 3)
            if random_move == 0:
                x -= 1
            elif random_move == 1:
                x += 1
            elif random_move == 2:
                y -= 1
            elif random_move == 3:
                y += 1
            # Check if the movement is legal.
            if x < 0 or x >= N or y < 0 or y >= N or matrix[x][y] == M:
                x = old_x
                y = old_y
                continue
            # Check how far away from the structure the particle is.
            particle_radius = (x - x_middle) ** 2 + (y - y_middle) ** 2
            # Abandon the particle if it strays too far from the structure.
            if particle_radius > kill_proportion * (structure_radius ** 2):
                break
            if is_near_structure(x, y, matrix, N, M):
                # If it's near, it has a probability p to stick to the structure.
                if (random.random() > p):
                    continue
                matrix[x][y] += 1
                # If we stuck M times, we add the particle to the stucture in this spot.
                if matrix[x][y] == M:
                    # Update the particle counter and the radius of the structure.
                    if particle_radius > (structure_radius ** 2):
                        structure_radius = math.sqrt(particle_radius)
                    i += 1
                break
    return structure_radius

# Initialize the simulation grid with a ball of given radius in the centre.
@jit(nopython=True)
def initialize_matrix(matrix, N, M, structure_radius):
    x_middle = N / 2
    y_middle = N / 2
    for x in range(0, N):
        for y in range(0, N):
            if (x - x_middle) ** 2 + (y - y_middle) ** 2 < (structure_radius ** 2):
                matrix[x][y] = M
            else :
                matrix[x][y] = 0

@jit(nopython=True)
def normalize_matrix(matrix, M):
    return np.where(matrix == M, 1, 0)

# Plot the current state of the system and save it to a buffer.
def plot_frame(matrix, M):
    data = normalize_matrix(matrix, M)
    fig, ax = plt.subplots(figsize=(11, 11)) 
    fig.tight_layout(pad=0)
    colors = ['#01031f', '#bed6ec']
    cmap = ListedColormap(colors)
    plt.imshow(data, cmap=cmap, interpolation='nearest', alpha=data.astype(float))
    plt.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.clf()
    plt.close()
    return Image.open(buffer)

def main(p, M, L, N, frames=150, structure_radius=3, kill_proportion=3):
    particles_per_frame = L // frames
    # Initialize simulation grid.
    matrix = np.zeros((N, N), dtype=np.int32)
    initialize_matrix(matrix, N, M, structure_radius)
    images = []
    images.append(plot_frame(matrix, M))

    # Simulate the evolution and plot it at regular intervals.
    i = 0
    while i < L:
        structure_radius = simulate_particles(matrix, N, M, structure_radius, kill_proportion, p, particles_per_frame)
        images.append(plot_frame(matrix, M))
        i += particles_per_frame
        print(i)

    imageio.mimsave('output.gif', images, duration=frames/30)

    return matrix
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLA simulation")
    parser.add_argument("--p", type=float, default=1, help="Probability of sticking to the structure")
    parser.add_argument("--M", type=int, default=1, help="Number of times we need to stick to get added to the structure")
    parser.add_argument("--L", type=int, default=30000, help="Number of particles")
    parser.add_argument("--N", type=int, default=800, help="Dimensions of the simulation grid")
    args = parser.parse_args()
    main(args.p, args.M, args.L, args.N, frames=120)
