"""Epsilon Greedy Exploration Strategy."""
import numpy as np
import random
import math
import torch
from tensordict.tensordict import TensorDict
from src.agents.dqn_agent import DQN
from src.environment.env import SumoEnvironment

class EpsilonGreedy(object):
    def __init__(self,
                 device: torch.device,
                 steps_done:int = 0,
                 EPS_START:float = 0.9,
                 EPS_END:float = 0.05,
                 EPS_DECAY:int = 1000
                 ) -> None:
        self.steps_done = steps_done
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.device = device

    def select_action(self, 
                      state: TensorDict, 
                      policy_net:DQN, 
                      env: SumoEnvironment
                      )-> torch.Tensor:
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = state.to(self.device)
                return policy_net(**state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], 
                                device=self.device, 
                                dtype=torch.long
            )


    
