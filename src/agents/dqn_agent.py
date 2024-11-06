import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import random
from tensordict.tensordict import TensorDict
import random
import numpy as np
from typing import Dict, Tuple, NamedTuple, List, Union, Literal
from types import NoneType

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

    def __len__(self)->int:
        return len(self.memory)

# Policy net
class DQN(nn.Module):
    def __init__(self, n_actions:int, CNN_out_dim:int, other_future_dim:int = 21):
        super(DQN, self).__init__()

        self.CNN = DQN._make_blocks(in_dim = 1, 
                                    out_dim = CNN_out_dim, 
                                    layer_type= 'CNN')
        self.Fusion = DQN._make_blocks(in_dim = CNN_out_dim + other_future_dim, 
                                       out_dim = n_actions, 
                                       layer_type = 'Linear')
        self.estimate_concat_dim = other_future_dim+CNN_out_dim

    @staticmethod
    def _make_blocks(in_dim:int,
                     out_dim:int,
                     layer_type: Literal['CNN','Linear']
                     )->nn.ModuleList:
        layers = []
        num_blocks = out_dim//(2**5) if layer_type == 'CNN' else (in_dim - out_dim)//(2**5)
        for i in range(0,num_blocks):
            layers += DQN._make_layer_type(in_channels = i*2**5,
                                           out_channels = (i+1)*2**5,
                                           layer_type = layer_type,
                                           start_in_dim = in_dim,
                                           end_out_dim = out_dim,
                                           is_final = False if i < num_blocks -1 else True)
        if layer_type == 'CNN':
            # layers.append(nn.AdaptiveAvgPool2d(output_size= (2,2)))
            layers.append(nn.Flatten())
        return nn.ModuleList(layers)
    
    @staticmethod
    def _make_layer_type(in_channels:int, 
                         out_channels:int, 
                         layer_type: Literal['CNN','Linear'],
                         start_in_dim:int,
                         end_out_dim:int,
                         is_final:bool
                         )->List[nn.Module]:
        if layer_type == 'CNN':
            return [nn.Conv2d(in_channels if in_channels != 0 else start_in_dim, 
                              out_channels if not is_final else end_out_dim, 
                              kernel_size = 3, stride = 1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size= 2, stride= 2),
                ]
        else:
            return [nn.Linear(start_in_dim - in_channels,end_out_dim)] if is_final else \
                    [nn.Linear(start_in_dim - in_channels,start_in_dim -out_channels), nn.ReLU()]

    
    def forward(self, 
                phase_id: torch.Tensor,
                min_green: torch.Tensor,
                density: torch.Tensor,
                queue: torch.Tensor,
                mapofCars: torch.Tensor
                )-> torch.Tensor:
                
        for layer in self.CNN:
            mapofCars = layer(mapofCars)

        total_features = torch.cat([phase_id, min_green, density, queue, \
                                    mapofCars], dim= -1)
        for mlp_layer in self.Fusion:
            total_features = mlp_layer(total_features)
        return total_features