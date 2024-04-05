import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LineEnvironment:
    def __init__(self, length=100):
        self.length = length

class Agent:
    def __init__(self, start_pos, mass, safe_radius, goal, max_vel, max_accel, weight, other_agent):
        self.start_pos = start_pos
        self.position = start_pos
        self.mass = mass
        self.safe_radius = safe_radius
        self.goal = goal
        self.velocity = 0
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.weight = weight
        self.other_agent = other_agent  # Reference to the other agent

        self.position_space = np.linspace(0, 100, 1000)
        self.velocity_space = np.linspace(-max_vel, max_vel, 100)
        self.acceleration_space = np.linspace(-max_accel, max_accel, 10)

        self.value_function = np.zeros((len(self.position_space), len(self.velocity_space)))
        self.policy = np.zeros_like(self.value_function)

    def solve_hjb(self):
        for i, pos in enumerate(self.position_space):
            for j, vel in enumerate(self.velocity_space):
                min_cost = float('inf')
                best_accel = 0
                for accel in self.acceleration_space:
                    cost = self.compute_cost(pos, vel, accel, self.other_agent.position)
                    # print(f"Acceleration: {accel}, Cost: {cost}")
                    if cost < min_cost:
                        min_cost = cost
                        best_accel = accel
                self.value_function[i, j] = min_cost
                self.policy[i, j] = best_accel
                # print(f"Best acceleration: {best_accel}, Min Cost: {min_cost}")

    def compute_cost(self, pos, vel, accel, other_agent_pos):
        next_pos = pos + vel + 0.5 * accel  # Update to consider the effect of acceleration on next position
        distance_to_goal = abs(next_pos - self.goal)
        cost = self.weight[0] * distance_to_goal

        # # Add cost for high velocity when near the goal
        # proximity_to_goal_threshold = 10
        # if distance_to_goal < proximity_to_goal_threshold:
        #     cost += (vel ** 2) * self.weight[1]

        # # Consider the effect of acceleration: positive acceleration should reduce cost if the agent is far from the goal
        # if distance_to_goal >= proximity_to_goal_threshold:
        #     cost -= 0.5 * abs(accel) * self.weight[2]  # Reward for accelerating towards the goal

        safe_distance_to_other_agent = self.weight[1]  # Define a safe distance
        if other_agent_pos is not None:
            distance_to_other_agent = abs(next_pos - other_agent_pos)
            if distance_to_other_agent < safe_distance_to_other_agent:
                # The closer the agents are, the higher the cost for high velocity
                velocity_penalty = (vel ** 2) / (safe_distance_to_other_agent - distance_to_other_agent) 
                cost += self.weight[2]*velocity_penalty

        # Penalty for safety radius violation
        if other_agent_pos is not None:
            distance_to_other_agent = abs(next_pos - other_agent_pos)
            if distance_to_other_agent < self.safe_radius:
                cost += self.weight[3]

        return cost

    def get_optimal_action(self, pos, vel):
        # Find nearest discretized state
        pos_idx = np.abs(self.position_space - pos).argmin()
        vel_idx = np.abs(self.velocity_space - vel).argmin()
        # Retrieve optimal action
        return self.policy[pos_idx, vel_idx]

class RobotAgent(Agent):
    pass

class HumanAgent(Agent):
    pass


def run_simulation_with_animation(num_runs=1, time_steps=100):
    env = LineEnvironment()
    for run in range(num_runs):
        robot = RobotAgent(start_pos=0, mass=2, safe_radius=5, goal=25, max_vel=10, max_accel=1, weight=[10,50,5,1000], other_agent=None)
        human = HumanAgent(start_pos=100, mass=1, safe_radius=5, goal=25, max_vel=40, max_accel=4, weight=[10,50,5,1000], other_agent=robot)
        robot.other_agent = human

        robot.solve_hjb()
        human.solve_hjb()

        fig, ax = plt.subplots()
        ax.set_xlim(0, env.length)
        ax.set_ylim(-20, 20)

        robot_dot, = ax.plot([], [], 'ro', label='Robot')
        human_dot, = ax.plot([], [], 'bo', label='Human')
        robot_radius = plt.Circle((robot.position, 0), robot.safe_radius, color='r', fill=False)
        human_radius = plt.Circle((human.position, 0), human.safe_radius, color='b', fill=False)
        ax.add_patch(robot_radius)
        ax.add_patch(human_radius)
        center_line = ax.axhline(y=0, color='gray', linestyle='--')  # Line in the middle
        robot_goal_dot = ax.plot(robot.goal, 0, 'yo', label='G_r')[0]
        human_goal_dot = ax.plot(human.goal, 0, 'go', label='G_H')[0]

        # Create text objects to display velocities
        robot_velocity_text = ax.text(0, 15, '', fontsize=9)
        human_velocity_text = ax.text(0, 10, '', fontsize=9)

        def init():
            robot_dot.set_data([], [])
            human_dot.set_data([], [])
            robot_radius.center = (0, 0)
            human_radius.center = (100, 0)
            robot_velocity_text.set_text('')
            human_velocity_text.set_text('')
            return robot_dot, human_dot, robot_radius, human_radius, robot_velocity_text, human_velocity_text

        def animate(i):
            # Calculate optimal action and update positions and velocities
            robot_acc = robot.get_optimal_action(robot.position, robot.velocity)
            human_acc = human.get_optimal_action(human.position, human.velocity)
            robot.velocity += robot_acc
            human.velocity += human_acc
            robot.position += robot.velocity
            human.position += human.velocity

            # Update visual elements
            robot_dot.set_data(robot.position, 0)
            human_dot.set_data(human.position, 0)
            robot_radius.center = (robot.position, 0)
            human_radius.center = (human.position, 0)
            robot_velocity_text.set_text(f'Rob_Vel: {robot.velocity:.2f}')
            human_velocity_text.set_text(f'Hum_Vel: {human.velocity:.2f}')

            # Recalculate HJB solution based on new positions
            robot.solve_hjb()
            human.solve_hjb()

            distance_between_agents = abs(robot.position - human.position)

            if distance_between_agents < robot.safe_radius + human.safe_radius or \
                (robot.position > human.position and robot.start_pos < human.start_pos) or \
                (human.position > robot.position and human.start_pos < robot.start_pos) or \
                abs(robot.position - robot.goal) < 1 or abs(human.position - human.goal) < 1:
                ani.event_source.stop()

            # if abs(robot.position - robot.goal) < 1 or abs(human.position - human.goal) < 1:
            #     ani.event_source.stop()

            return robot_dot, human_dot, robot_radius, human_radius, robot_velocity_text, human_velocity_text

        ani = FuncAnimation(fig, animate, init_func=init, frames=time_steps, interval=100, blit=True)
        plt.legend()
        plt.show()


# Example usage
run_simulation_with_animation(num_runs=10, time_steps=200)

# Simulation would use get_optimal_action to determine actions at each step

