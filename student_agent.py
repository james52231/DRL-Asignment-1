# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("taxi_table.pkl", "rb") as f:
q_table_file = pickle.load(f)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
        # Path to the Q-table file
    q_table_file = "taxi_qtable.pkl"
    
    # Check if Q-table exists and load it
    if os.path.exists(q_table_file):
        try:
            with open(q_table_file, 'rb') as f:
                q_table = pickle.load(f)
        except Exception:
            # If loading fails, use fallback strategy
            return fallback_strategy(obs)
    else:
        # If Q-table doesn't exist, use fallback strategy
        return fallback_strategy(obs)
    
    # Create a state key from the observation
    state_key = create_state_key(obs)
    
    # If state is not in Q-table, use fallback strategy
    if state_key not in q_table:
        return fallback_strategy(obs)
    
    # Choose the action with the highest Q-value
    return np.argmax(q_table[state_key])

def create_state_key(obs):
    """
    Create a compact state representation from the observation to use as a key for the Q-table.
    
    Args:
        obs: The observation tuple from the environment
    
    Returns:
        tuple: A simplified state representation
    """
    taxi_row, taxi_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # Create a more compact state representation
    # This is important to reduce the size of the Q-table
    state_key = (
        taxi_row, taxi_col,
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,
        passenger_look, destination_look
    )
    
    return state_key

def fallback_strategy(obs):
    """
    Provide a reasonable action when the Q-table doesn't have an entry for the current state.
    
    Args:
        obs: The observation tuple from the environment
    
    Returns:
        int: The action to take (0-5)
    """
    taxi_row, taxi_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # Stations positions
    stations = [(s0_row, s0_col), (s1_row, s1_col), (s2_row, s2_col), (s3_row, s3_col)]
    
    # If passenger is visible, try to pick up
    if passenger_look == 1:
        return 4  # Pickup action
    
    # If destination is visible and we assume passenger is picked up, try to drop off
    if destination_look == 1:
        return 5  # Dropoff action
    
    # Move toward open directions (avoid obstacles)
    possible_moves = []
    if obstacle_north == 0:
        possible_moves.append(1)  # Move North
    if obstacle_south == 0:
        possible_moves.append(0)  # Move South
    if obstacle_east == 0:
        possible_moves.append(2)  # Move East
    if obstacle_west == 0:
        possible_moves.append(3)  # Move West
    
    if possible_moves:
        return random.choice(possible_moves)



    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
    #test
