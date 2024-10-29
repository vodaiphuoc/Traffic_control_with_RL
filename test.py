from src.environment.env import SumoEnvironment
import torch
from collections import namedtuple, deque
from tensordict.tensordict import TensorDict
import random
import numpy as np
from typing import Dict, Tuple, NamedTuple, List, Union
from src.agents.dqn_agent import DQN


image = torch.rand(size=(1,1,60,60))
CNN_block = DQN._make_blocks(in_dim= 1, out_dim= 128, layer_type= 'CNN')

for layer in CNN_block:
    image = layer(image)
    print(layer, image.shape)

