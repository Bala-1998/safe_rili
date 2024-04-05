def run_simulation(grid_world, value_iteration, num_runs=100):
    # Statistics
    stats = {
        'collisions': 0,
        'robot_reached_goal_first': 0,
        'human_reached_goal_first': 0,
        'safety_radius_changes': 0
    }

    for run in range(num_runs):
        # Reset positions for each run
        grid_world.set_positions(robot_pos=(1, 1), human_pos=(9, 0), robot_goal=(9, 0), human_goal=(0, 0))
        
        robot_reached_goal = False
        human_reached_goal = False
        while not robot_reached_goal and not human_reached_goal:
            robot_action = value_iteration.policy[grid_world.robot_pos]
            if robot_action:
                grid_world.robot_pos = value_iteration.get_next_state(grid_world.robot_pos, robot_action)
                if grid_world.robot_pos == grid_world.robot_goal:
                    robot_reached_goal = True
                elif grid_world.robot_pos == grid_world.human_pos:
                    stats['collisions'] += 1
                    break  # Collision occurred

            # Human's predetermined movement (can be made more dynamic)
            # For now, we assume the human stays static
            # grid_world.human_pos = next_human_position

            if grid_world.human_pos == grid_world.human_goal:
                human_reached_goal = True

        if robot_reached_goal:
            stats['robot_reached_goal_first'] += 1
        elif human_reached_goal:
            stats['human_reached_goal_first'] += 1

        # Safety radius logic (can be expanded based on specific criteria)
        # stats['safety_radius_changes'] += change_in_safety_radius

    return stats

# Run the simulation and collect data
simulation_stats = run_simulation(grid_world, value_iteration)
simulation_stats

# Plotting the collected statistics

# Number of Collisions
plt.figure(figsize=(10, 4))
plt.bar(['Collisions', 'Robot Reached Goal', 'Human Reached Goal'], 
        [simulation_stats['collisions'], simulation_stats['robot_reached_goal_first'], simulation_stats['human_reached_goal_first']])
plt.ylabel('Number of Occurrences')
plt.title('Simulation Statistics')
plt.show()

# Number of Safety Radius Changes (in this simulation it's always 0, as the human is static)
plt.figure(figsize=(5, 4))
plt.bar(['Safety Radius Changes'], [simulation_stats['safety_radius_changes']])
plt.ylabel('Number of Changes')
plt.title('Safety Radius Changes')
plt.show()
