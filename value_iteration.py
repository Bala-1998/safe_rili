import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

# Make sure to run the value iteration for multiple runs
# Have a strict collision check for the robot, the robot cannot collide with the human
# Have an action to reduce the safety radius for the robot if it has a positive reward
# Print statements for actions, states, rewards


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

    def human_move_towards_goal(self):
        if self.human_pos[0] > self.human_goal[0]:
            next_pos = (self.human_pos[0] - 1, self.human_pos[1])
        elif self.human_pos[0] < self.human_goal[0]:
            next_pos = (self.human_pos[0] + 1, self.human_pos[1])
        else:
            next_pos = self.human_pos

        if abs(next_pos[0] - self.robot_pos[0]) <= 1 and abs(next_pos[1] - self.robot_pos[1]) <= 1:
            return self.human_pos
        else:
            return next_pos

class ValueIteration:
    def __init__(self, grid_world, discount_factor=0.9, theta=0.1):
        self.grid_world = grid_world
        self.grid_size = grid_world.size
        self.states = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self.actions = ["up", "down", "left", "right", "stay"]
        self.discount_factor = discount_factor
        self.theta = theta  # Threshold for stopping the iteration
        self.value_map = np.zeros((self.grid_size, self.grid_size))
        self.policy = dict()

    def is_valid_state(self, state):
        x, y = state
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_next_state(self, state, action):
        x, y = state
        if action == "up":
            next_state = (x - 1, y)
        elif action == "down":
            next_state = (x + 1, y)
        elif action == "left":
            next_state = (x, y - 1)
        elif action == "right":
            next_state = (x, y + 1)
        else:
            next_state = (x,y)  # Stay in the current position if action is not recognized

        self.grid_world.human_pos = self.grid_world.human_move_towards_goal(self.grid_world)
        # Return the current state if moving to next_state would collide with the human or is invalid
        if not self.is_valid_state(next_state) or next_state == self.grid_world.human_pos:
            return state
        return next_state

    def reward(self, current_state, next_state):
        if next_state == self.grid_world.robot_goal:
            return 100  # Goal reward
        if next_state == self.grid_world.human_pos:
            return -10000  # Collision penalty
        # Safe distance check
        if abs(next_state[0] - self.grid_world.human_pos[0]) <= self.grid_world.robot_radius and \
           abs(next_state[1] - self.grid_world.human_pos[1]) <= self.grid_world.robot_radius:
            return -10
        return -1  # Movement cost

    def run_value_iteration(self, num_runs=1, max_iterations=1000):
        stats = {
            'collisions': 0,
            'robot_reached_goal_first': 0,
            'human_reached_goal_first': 0,
            'safety_radius_changes': 0
        }

        for run in range(num_runs):
            while True:
                delta = 0
                for state in self.states:
                    if state in [self.grid_world.robot_goal, self.grid_world.human_pos]:
                        continue
                    v = self.value_map[state]
                    action_values = []
                    for action in self.actions:
                        next_state = self.get_next_state(state, action)
                        reward = self.reward(state, next_state)
                        action_values.append(reward + self.discount_factor * self.value_map[next_state])
                    max_value = max(action_values)
                    self.value_map[state] = max_value
                    delta = max(delta, abs(v - max_value))
                    print("delta:", delta)
                if delta < self.theta:
                    print("inside theta")
                    break
            
            for state in self.states:
                if state in [self.grid_world.robot_goal, self.grid_world.human_pos]:
                    self.policy[state] = None
                else:
                    self.policy[state] = self.actions[np.argmax([self.reward(state, self.get_next_state(state, a)) +
                                                                self.discount_factor * self.value_map[self.get_next_state(state, a)] for a in self.actions])]


            self.grid_world.set_positions(robot_pos=(1, 1), human_pos=(9, 0), robot_goal=(9, 0), human_goal=(0, 0))
            robot_reached_goal = False
            human_reached_goal = False
            while not robot_reached_goal and not human_reached_goal:
                robot_action = self.policy[self.grid_world.robot_pos]
                if robot_action:
                    self.grid_world.robot_pos = self.get_next_state(self.grid_world.robot_pos, robot_action)
                    print("robot_pos:", self.grid_world.robot_pos)
                    if self.grid_world.robot_pos == self.grid_world.robot_goal:
                        robot_reached_goal = True
                    elif self.grid_world.robot_pos == self.grid_world.human_pos:
                        stats['collisions'] += 1
                        break

                self.grid_world.human_pos = self.grid_world.human_move_towards_goal()
                print("human_pos:", self.grid_world.human_pos)
                if self.grid_world.human_pos == self.grid_world.human_goal:
                    human_reached_goal = True

            if robot_reached_goal:
                stats['robot_reached_goal_first'] += 1
            elif human_reached_goal:
                stats['human_reached_goal_first'] += 1
            
            print(stats)


# Initializing grid world and value iteration
grid_world = GridWorld()
grid_world.set_positions(robot_pos=(1, 1), human_pos=(9, 0), robot_goal=(9, 0), human_goal=(0, 0))
value_iteration = ValueIteration(grid_world)
value_iteration.run_value_iteration(num_runs = 1)



