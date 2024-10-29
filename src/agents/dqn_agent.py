import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import random
from tensordict.tensordict import TensorDict
import random
import numpy as np
from typing import Dict, Tuple, NamedTuple, List, Union

class Transition(NamedTuple):
    state: TensorDict
    action: int
    next_state: TensorDict
    reward: float

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    @staticmethod
    def _toTensorDict(state: Dict[str, np.ndarray])->TensorDict:
        return TensorDict(source= {k: np.expand_dims(v,0) 
                                   for k,v in state.items()}, 
                          batch_size= 1)
    
    def push(self, 
             state: Dict[str,np.ndarray],
             action: torch.Tensor,
             next_state: Dict[str,np.ndarray],
             reward: Union[float, torch.Tensor])->None:
        self.memory.append(Transition(ReplayMemory._toTensorDict(state), 
                                      action, 
                                      ReplayMemory._toTensorDict(next_state),
                                      torch.tensor([reward]) if isinstance(reward,float) else reward
                            ))
                            
    def sample(self, batch_size)->List[Transition]:
        return random.sample(self.memory, batch_size)
    
    @staticmethod
    def get_batch(batch: List[Transition])->Tuple[torch.Tensor]:
        # converts batch-array of Transitions
        # to Transition of batch-arrays
        transition_batchs = Transition(*zip(*batch))
        return (torch.cat(transition_batchs.state),
                torch.cat(transition_batchs.action),
                torch.cat(transition_batchs.next_state),
                torch.cat(transition_batchs.reward)
                )

    def __len__(self)->int:
        return len(self.memory)

# Policy net
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


