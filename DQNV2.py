"""
All done by
Sajad Salavatidezfouli
"""
#%% libararies
import torch
import torch.nn as nn
import torch.optim as optim
import random
import ansys.fluent.core as pyfluent
import re
import numpy as np
import csv
import time as TIME

# Printing the console in a file
# from IPython.utils import io
# %logstart -o

#%% Functions

def process_probe_data(points, variable_type, file_prefix):
    values = {}

    # Loop through each point
    for point in points:
        # Generate the file
        solver_session.tui.report.surface_integrals(
            "vertex-avg",
            point,
            "()",
            variable_type,
            "yes",
            f"C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/{point}{file_prefix}.srp",
        )
        
        # Read the file and extract the last value
        with open(f"C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/{point}{file_prefix}.srp", 'r') as file:
            content = file.read()
        
        # Use regular expression to find the last value
        matches = re.findall(fr'{point}\s+([\d.]+)', content)
        
        # Check if matches are found
        if matches:
            value = float(matches[-1])  # Extract the last value
            values[point] = value
        else:
            print(f"Value not found for {point}.")
        
    return values



def combine_probe_data(points, pointsV, temperature_values, pressure_values, velocity_values):
    # Create an empty NumPy array
    states = np.empty((len(points) + len(points) + len(pointsV),))

    # Fill the array with temperature and pressure values
    for i, point in enumerate(points):
        states[i] = temperature_values.get(point, np.nan)
        states[i + len(points)] = pressure_values.get(point, np.nan)

    # Fill the array with velocity values
    for i, pointV in enumerate(pointsV):
        states[i + 2 * len(points)] = velocity_values.get(pointV, np.nan)
        
    return states



def calculate_surface_average_temperature():
    solver_session.tui.report.surface_integrals.area_weighted_avg(
        "bottom",
        "()",
        "temperature",
        "yes",
        "C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/aveT.srp",
    )
        
    with open("C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/aveT.srp", 'r') as file:
        content = file.read()

    # Use regular expression to find the last temperature value
    matches = re.findall(r'bottom\s+([\d.]+)', content)

    # Check if matches are found
    if matches:
        surface_average_T = float(matches[-1])
        return surface_average_T
    else:
        print("Value not found in the file.")
        return None



def apply_action(action):
    action_value = 0.1 * (action + 1)  # Convert action index to actual action value
    
    # Code to apply the action in the CFD environment ...
    solver_session.tui.define.boundary_conditions.velocity_inlet(
        "inlet",
        "no",
        "no",
        "yes",
        "yes",
        "no",
        str(action_value),
        "no",
        "0",
        "no",
        "0.5",
        "no",
        "no",
        "yes",
        "5",
        "10",
    )
    
    # Run for 10 timesteps
    solver_session.tui.solve.dual_time_iterate(
        "10",  # number of time steps
        "40",  # some default number
    )


#%% Define the Replay Memory class
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

#%% Define the DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_memory_capacity, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.batch_size = batch_size

    def select_action(self, state, epsilon):
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state).float())
                return torch.argmax(q_values).item()
            
    def update(self, state, action, next_state, reward, done):
        self.replay_memory.push((state, action, next_state, reward, done))

        if len(self.replay_memory.memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)
        batch = zip(*transitions)
        state_batch, action_batch, next_state_batch, reward_batch = map(torch.tensor, batch)
        # state_batch = next_state_batch === [batch_size, state_dim]
        # action_batch === [batch_size,]
        # reward_batch === [batch_size,]
        # done_batch   === [batch_size,]
        

        self.optimizer.zero_grad()
        q_values = self.q_network(state_batch.float())  # because input is [batch_size, state_dim], output is [batch_size, action_dim]
        # q_values = [batch_size, action_dim]

        target_q_values = q_values.clone().detach()
        # target_q_values = [batch_size, action_dim]
        next_q_values = self.q_network(next_state_batch.float())
        # next_q_values = [batch_size, action_dim]
        next_q_max = torch.max(next_q_values, dim=1)[0]
        # next_q_values [batch_size,]
        target_q_values[torch.arange(self.batch_size), action_batch] = reward_batch + 0.99 * next_q_max * (1 - done_batch.float())
        # this "target_q_values[torch.arange(self.batch_size), action_batch]" only updates the pair of 
        #   torch.arange(self.batch_size) and action_batch and not the whole traget_q_values
        
        loss = self.loss_function(q_values, target_q_values)
        # we are seeking to equalize the q-values (from the nework) to the q-vlaues (from the bellman equation, i.e. target-q-values)
        # Bellman q-values are optimum
        
        #The Bellman optimality equation states that 
        #    the optimal Q-value for a state-action pair is equal to the immediate reward plus
        #    the discounted value of the maximum Q-value in the next state. 
        
        
        loss.backward()
        self.optimizer.step()

#%% Reward calculation
def compute_reward(surface_average_temp):
    desired_temp = 1
    threshold = 0.05

    deviation = abs(surface_average_temp - desired_temp)
    if deviation <= threshold:
        return 1
    else:
        # Compute the ramped reward
        reward = 0.1 - (deviation - threshold) * 0.1
        return reward

#%% Instantiate the DQN agent
state_dim = 22  # Specify the dimensionality of the state
action_dim = 10  # Specify the number of possible actions
replay_memory_capacity = 100
batch_size = 10
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim,
                 replay_memory_capacity=replay_memory_capacity, batch_size=batch_size)

num_episodes = 50
max_steps_per_episode = 10000
dt_size = 0.1 # no automatic set. manually set in FLUENT
n_dt_per_step = 10 # this is defined in one the apply_action function above
#%% Loading the previous saved episodes, if exist

# Load the saved data for the desired episode
episode_to_load = 99  # Specify the episode number you want to load
start = 0

try:
    saved_data = torch.load(f'episode_{episode_to_load}.pth')
    agent.q_network.load_state_dict(saved_data['network_state'])  # Load the network parameters
    agent.optimizer.load_state_dict(saved_data['optimizer_state'])  # Load the optimizer state
    agent.replay_memory = saved_data['replay_memory']  # Set the replay memory
    start = episode_to_load + 1

except FileNotFoundError:
    # Handle the case when the file does not exist, e.g., initialize the agent randomly or with default values
    agent_state = None  # Replace None with your preferred way to initialize the agent

    
#%%Training loop


# Create a CSV file and write the header
with open('episode_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'time', 'action', 'surface_average_T', 'reward', 'total_reward'])


    for episode in range(100):
        # Running FLUENT code and Reset the CFD environment
        solver_session = pyfluent.launch_fluent(mode="solver", show_gui="True", case_file_name="C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/Fluent.cas")
        # solver_session = pyfluent.launch_fluent(mode="solver", case_filepath="C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/Fluent.cas")
        solver_session.health_check_service.is_serving
        solver_session.tui.file.read_data('C:/Users/SSD/Desktop/Saeid/DQN-ReadyToRun/Fluent.dat')
        
        ######## Get the initial state from CFD environment
        points = ['point0', 'point1', 'point2', 'point3', 'point4', 'point5', 'point6', 'point7', 'point8', 'point9']
        pointsV = ['pointup1', 'pointup2']
        temperature_values = process_probe_data(points, "temperature", "T")
        pressure_values = process_probe_data(points, "pressure", "P")
        velocity_values = process_probe_data(pointsV, "velocity-magnitude", "V")
        states = combine_probe_data(points, pointsV, temperature_values, pressure_values, velocity_values)
        state = states
        ########
    
        total_reward = 0
        
        time = 0
    
        for step in range(max_steps_per_episode):
            epsilon = max(0.1, 1 - episode / num_episodes)
            action = agent.select_action(state, epsilon)
    
            # Apply the action to the CFD environment
            apply_action(action)
    
            ######## Get the next state from the CFD environment
            points = ['point0', 'point1', 'point2', 'point3', 'point4', 'point5', 'point6', 'point7', 'point8', 'point9']
            pointsV = ['pointup1', 'pointup2']
            temperature_values = process_probe_data(points, "temperature", "T")
            pressure_values = process_probe_data(points, "pressure", "P")
            velocity_values = process_probe_data(pointsV, "velocity-magnitude", "V")
            states = combine_probe_data(points, pointsV, temperature_values, pressure_values, velocity_values)
            next_state = states
            ###########
            
            # reward calculation
            surface_average_T = calculate_surface_average_temperature()
            reward = compute_reward(surface_average_T)  # Compute the reward based on surface average temperature
            done = False
    
            agent.update(state, action, next_state, reward, done)
    
            state = next_state
            total_reward += reward
    
            if done:
                break

            writer.writerow([episode, time, action, surface_average_T, reward, total_reward])
            file.flush()  # Flush the buffer to ensure immediate write
            time += n_dt_per_step * dt_size
        
        
        
        #close FLUENT session at the end of each episode
        solver_session.exit()
        # wait 10 seconds until the FLUENT is completely close
        TIME.sleep(10)
        
        # Save the agent's state and other necessary information
        network_state = agent.q_network.state_dict()  # Get the network parameters
        optimizer_state = agent.optimizer.state_dict()  # Get the optimizer state
        replay_memory = agent.replay_memory  # Get the replay memory
        
        data_to_save = {
            'episode': episode,
            'network_state': network_state,
            'optimizer_state': optimizer_state,
            'replay_memory': replay_memory,
            # Include other necessary information that you may need to continue training
        }
        torch.save(data_to_save, f'episode_{episode}.pth')















