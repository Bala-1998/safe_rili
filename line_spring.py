import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LineEnvironment:
    def __init__(self):
        self.length = 120  # Arbitrary length of the line environment

class Agent:
    def __init__(self, start_pos, mass, safe_radius, goal, max_vel, other_agent=None):
        self.position = start_pos
        self.mass = mass
        self.safe_radius = safe_radius
        self.goal = goal
        self.velocity = 0
        self.max_vel = max_vel
        self.other_agent = other_agent

class RobotAgent(Agent):
    def __init__(self, start_pos, mass, safe_radius, goal, max_vel, omega_n, zeta, k_goal, other_agent=None):
        super().__init__(start_pos, mass, safe_radius, goal, max_vel, other_agent)
        self.AR = np.array([[0, 1], [-omega_n**2, -2*zeta*omega_n]])  # Dynamics matrix AR
        self.BR = np.array([[0], [k_goal/mass]])  # Control input matrix BR
        self.k_hat = safe_radius  # Interaction constant
        self.s = 0.01  # Scaling factor

    def get_state_derivative(self):
        # Ensure self.position and self.velocity are scalars
        position = np.array(self.position).reshape(-1)
        velocity = np.array(self.velocity).reshape(-1)

        xR = np.array([position, velocity])  # State vector for the robot
        goal_diff = np.array([self.goal - self.position]).reshape(-1)
        uR = goal_diff  # Control input towards the goal
        xH = np.array([[self.other_agent.position], [self.other_agent.velocity]])  # State of the human
        interaction_term = self.k_hat * self.s * (xH - xR)  # Interaction term
        dyn_term = np.dot(self.AR, xR)
        cont_term = np.dot(self.BR, uR)
        print("dyn_term_robot:", dyn_term)
        print("cont_term_robot:", cont_term)
        print("interaction_term_robot:", interaction_term)
        vel = dyn_term[1,0] + cont_term[1] - interaction_term[0,0]
        velocity_value = vel # Access the scalar velocity value

        # Limiting the velocity to max_vel
        if velocity_value > self.max_vel:
            velocity_value = self.max_vel
        return velocity_value

class HumanAgent(Agent):
    def __init__(self, start_pos, mass, safe_radius, goal, max_vel, omega_n, zeta, k_goal, other_agent=None):
        super().__init__(start_pos, mass, safe_radius, goal, max_vel, other_agent)
        self.AH = np.array([[0, 1], [omega_n**2, -2*zeta*omega_n]])  # Dynamics matrix AH
        self.BH = np.array([[0], [k_goal/mass]])  # Control input matrix BH
        self.k = safe_radius  # Interaction constant
        self.s = 0.001  # Scaling factor

    def get_state_derivative(self):
        # Ensure self.position and self.velocity are scalars
        position = np.array(self.position).reshape(-1)
        velocity = np.array(self.velocity).reshape(-1)

        xH = np.array([position, velocity])  # State vector for the robot
        goal_diff = np.array([self.goal - self.position]).reshape(-1)
        uH = goal_diff  # Control input towards the goal
        xR = np.array([[self.other_agent.position], [self.other_agent.velocity]])  # State of the robot
        interaction_term = self.k * self.s * (xR - xH)  # Interaction term
        dyn_term = np.dot(self.AH, xH)
        cont_term = np.dot(self.BH, uH)
        print("dyn_term_human:", dyn_term)
        print("cont_term_human:", cont_term)
        print("interaction_term_human:", interaction_term)
        vel = dyn_term[1,0] + cont_term[1] - interaction_term[0,0]
        velocity_value = vel # Access the scalar velocity value

        # Limiting the velocity to max_vel
        if velocity_value > self.max_vel:
            velocity_value = self.max_vel
        return velocity_value



# Function to run simulation with animation
def run_simulation_with_animation(num_runs=1, time_steps=100):
    env = LineEnvironment()
    for run in range(num_runs):
        robot = RobotAgent(start_pos=30, mass=1, safe_radius=1, goal=50, max_vel=10, omega_n = 2.0, zeta= 0.7, k_goal = 10,  other_agent = None)
        human = HumanAgent(start_pos=70, mass=1, safe_radius=1, goal=50, max_vel=10, omega_n = 2.0, zeta= 0.7, k_goal = 10, other_agent = robot)
        robot.other_agent = human

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
        
        # Animate function to update the positions and velocities
        def animate(i):
            # Update state derivatives
            robot_velocity = robot.get_state_derivative()
            print("robot_velocity:", robot_velocity)
            human_velocity = human.get_state_derivative()
            print("human_velocity:", human_velocity)

            # Simple Euler integration for demonstration purposes
            robot.position += robot_velocity*0.01
            human.position += human_velocity*0.01

            # Update visual elements
            robot_dot.set_data(robot.position, 0)
            human_dot.set_data(human.position, 0)
            robot_radius.center = (robot.position, 0)
            human_radius.center = (human.position, 0)
            robot_velocity_text.set_text(f'Rob_Vel: {robot_velocity:.2f}')
            human_velocity_text.set_text(f'Hum_Vel: {human_velocity:.2f}')

            return robot_dot, human_dot, robot_radius, human_radius, robot_velocity_text, human_velocity_text


        ani = FuncAnimation(fig, animate, init_func=init, frames=time_steps, interval=100, blit=True)
        plt.legend()
        plt.show()

# Run the simulation
run_simulation_with_animation(num_runs=10, time_steps=100)

# Simulation would use get_optimal_action to determine actions at each step

