import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size))
        # Icons: 0 - empty, 1 - robot, 2 - human, 3 - robot's goal, 4 - human's goal
        self.robot_pos = None
        self.human_pos = None
        self.robot_goal = None
        self.human_goal = None
        self.robot_radius = 2
        self.human_radius = 1

    def add_obstacle(self, position):
        self.grid[position] = -1

    def set_positions(self, robot_pos, human_pos, robot_goal, human_goal):
        self.robot_pos = robot_pos
        self.human_pos = human_pos
        self.robot_goal = robot_goal
        self.human_goal = human_goal
        self.update_grid()

    def update_grid(self):
        self.grid.fill(0)  # Clear grid
        self.grid[self.robot_pos] = 1
        self.grid[self.human_pos] = 2
        self.grid[self.robot_goal] = 3
        self.grid[self.human_goal] = 4

    def visualize(self):
        cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(self.grid, cmap=cmap, norm=norm)

        # Creating a legend
        legend_elements = [mpatches.Patch(color='red', label='Robot'),
                           mpatches.Patch(color='blue', label='Human'),
                           mpatches.Patch(color='green', label="Robot's Goal"),
                           mpatches.Patch(color='yellow', label="Human's Goal")]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.show()

# Example usage
grid_world = GridWorld()
grid_world.set_positions(robot_pos=(1, 1), human_pos=(9, 0), robot_goal=(9, 0), human_goal=(0, 0))
grid_world.visualize()

