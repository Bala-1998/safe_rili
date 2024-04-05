import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LineEnvironment:
    def __init__(self, length=100):
        self.length = length

class Agent:
    def __init__(self, name, start_pos, safe_radius, goal, velocity=1):
        self.name = name
        self.position = start_pos
        self.safe_radius = safe_radius
        self.goal = goal
        self.velocity = velocity

    def move_towards_goal(self):
        if self.position < self.goal:
            self.position += self.velocity
        elif self.position > self.goal:
            self.position -= self.velocity

    def evade(self, other):
        if self.position < other.position:
            self.position -= self.velocity / 2
        else:
            self.position += self.velocity / 2

def animate_interaction_restart(env, robot, human, goal_threshold=5, max_steps=100):
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.length)
    ax.set_ylim(-10, 10)

    central_line_y = 0
    robot_dot, = ax.plot([], [], 'ro', label='Robot')
    human_dot, = ax.plot([], [], 'bo', label='Human')
    robot_circle = plt.Circle((robot.position, central_line_y), robot.safe_radius, color='r', fill=False)
    human_circle = plt.Circle((human.position, central_line_y), human.safe_radius, color='b', fill=False)
    ax.add_patch(robot_circle)
    ax.add_patch(human_circle)
    robot_goal_dot = ax.plot(robot.goal, central_line_y, 'yo', label='G_r')[0]
    human_goal_dot = ax.plot(human.goal, central_line_y, 'go', label='G_H')[0]

    positions = {'robot': [], 'human': [], 'robot_radius': [], 'human_radius': []}

    for step in range(max_steps):
        distance = abs(robot.position - human.position)
        if distance <= robot.safe_radius or distance <= human.safe_radius:
            robot.evade(human)
            human.evade(robot)
        else:
            robot.move_towards_goal()
            human.move_towards_goal()

        positions['robot'].append(robot.position)
        positions['human'].append(human.position)
        positions['robot_radius'].append(robot.safe_radius)
        positions['human_radius'].append(human.safe_radius)

        if abs(robot.position - robot.goal) < goal_threshold or abs(human.position - human.goal) < goal_threshold:
            break

    def update(frame):
        robot_dot.set_data(positions['robot'][frame], central_line_y)
        human_dot.set_data(positions['human'][frame], central_line_y)
        robot_circle.set_radius(positions['robot_radius'][frame])
        robot_circle.center = (positions['robot'][frame], central_line_y)
        human_circle.set_radius(positions['human_radius'][frame])
        human_circle.center = (positions['human'][frame], central_line_y)
        return robot_dot, human_dot, robot_circle, human_circle

    ani = FuncAnimation(fig, update, frames=len(positions['robot']), blit=True, repeat=False)
    plt.legend()
    plt.show()

def run_simulation_with_restarts(num_runs=5, robot_goal=None, human_goal=None):
    env = LineEnvironment()
    for _ in range(num_runs):
        robot_safe_radius = random.uniform(3, 10)
        robot_goal = robot_goal if robot_goal is not None else random.uniform(0, env.length)
        human_goal = human_goal if human_goal is not None else random.uniform(0, env.length)
        robot = Agent("Robot", 100, robot_safe_radius, robot_goal)
        human = Agent("Human", 0, 4, human_goal)
        animate_interaction_restart(env, robot, human)

# Example usage
#run_simulation_with_restarts()  # Random goals with restarts
run_simulation_with_restarts(num_runs=5, robot_goal=50, human_goal=50)  # User-specified goals with restarts
