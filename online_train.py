# -*- coding: utf-8 -*-
import math
import random
import matplotlib
import matplotlib.pyplot as plt

from itertools import count
from src.environment.env import SumoEnvironment
from src.agents.dqn_agent import DQN, ReplayMemory
from src.exploration.epsilon_greedy import EpsilonGreedy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
map_config_path= "src/nets/2way-single-intersection/map_config.json"
net_file="src/nets/2way-single-intersection/single-intersection.net.xml"
route_file="src/nets/2way-single-intersection/single-intersection-vhvh.rou.xml"
out_csv_name="outputs/2way-single-intersection/dqn"



env = SumoEnvironment(
    map_config_path = os.path.join(current_file_path,map_config_path),
    net_file = os.path.join(current_file_path,net_file),
    route_file = os.path.join(current_file_path,route_file),
    out_csv_name = os.path.join(current_file_path,out_csv_name),
    single_traffic_light=True,
    use_gui=False,
    num_seconds=50000,
)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
state = ReplayMemory._toTensorDict(state)
n_observations = len(state)

policy_net = DQN(n_actions = n_actions, 
                 CNN_out_dim = 128, 
                 other_future_dim = 21).to(device)

target_net = DQN(n_actions = n_actions,
                 CNN_out_dim = 128,
                 other_future_dim = 21).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000)

action_agent = EpsilonGreedy(device = device,
                            EPS_START = EPS_START,
                            EPS_END = EPS_END,
                            EPS_DECAY = EPS_DECAY
                            )


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network is updated at every step with a 
# `soft update controlled by the hyperparameter ``TAU``, which was previously defined.

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = ReplayMemory.get_batch(batch= transitions)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_batch = state_batch.to(device)
    state_action_values = policy_net(**state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        non_final_next_states = non_final_next_states.to(device)
        next_state_values[non_final_mask] = target_net(**non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.to(device)
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

######################################################################
# Below, you can find the main training loop. At the beginning we reset
# the environment and obtain the initial ``state`` Tensor. Then, we sample
# an action, execute it, observe the next state and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 60
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    print("Episode:", i_episode)
    # Initialize the environment and get its state
    state, info = env.reset()
    state = ReplayMemory._toTensorDict(state)

    for t in tqdm(count()):
        action = action_agent.select_action(state,policy_net,env)

        observation, reward, terminated, truncated, infor = env.step(action.item())
        
        done = terminated or truncated
        next_state = None if terminated else observation
        
        # Store the transition in memory
        next_state = memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

