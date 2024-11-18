from typing import Dict, Tuple, NamedTuple, List, Union, Literal
from tensordict.tensordict import TensorDict
from collections import deque
from types import NoneType

import torch
import numpy as np
import random

class Transition(NamedTuple):
    state: TensorDict
    action: int
    next_state: TensorDict
    reward: float

class ReplayMemory(object):
    def __init__(self, capacity:int):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    @staticmethod
    def _toTensorDict(state: Dict[str, np.ndarray])->TensorDict:
        return TensorDict(source= {k: np.expand_dims(v,0).astype(np.float32)
                                   for k,v in state.items()},
                          batch_size= 1)
    
    def push(self,
             state: TensorDict,
             action: torch.Tensor,
             next_state: Union[Dict[str,np.ndarray], NoneType],
             reward: Union[float, torch.Tensor])->None:
        """Append to Relay memory"""
        assert isinstance(state, TensorDict), f"Found type {type(state)}"
        next_state = ReplayMemory._toTensorDict(next_state)
        self.memory.append(Transition(state, 
                                      action, 
                                      next_state,
                                      torch.tensor([reward]) if isinstance(reward,float) else reward
                            ))
        return next_state
                            
    def sample(self, batch_size)->List[Transition]:
        return random.sample(self.memory, batch_size)
    
    @staticmethod
    def get_batch(batch: List[Transition])->Tuple[torch.Tensor]:
        # converts batch-array of Transitions
        # to Transition of batch-arrays
        transition_batchs = Transition(*zip(*batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          transition_batchs.next_state)), 
                                          device=batch[0].action.device, 
                                          dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in transition_batchs.next_state
                                                    if s is not None])
        
        return (torch.cat(transition_batchs.state).to(torch.float32),
                torch.cat(transition_batchs.action),
                torch.cat(transition_batchs.reward),
                non_final_mask,
                non_final_next_states
                )
    
    @property
    def is_full(self):
        return self.__len__ == self.capacity

    def __len__(self)->int:
        return len(self.memory)
