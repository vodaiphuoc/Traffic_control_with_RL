from typing import Dict, Tuple, List, Union, Literal
from tensordict.tensordict import TensorDict
import numpy as np
import torch
from collections import namedtuple

# super class
Base_Transition = namedtuple('Base_Transition',['state','action'])

class Transition(Base_Transition):
	def __new__(cls,
				state: Union[TensorDict, Tuple[TensorDict]],
				action: Union[torch.Tensor, Tuple[torch.Tensor]]
				)->Base_Transition:
		if isinstance(state, tuple) and isinstance(action, tuple):
			return super(Transition,cls).__new__(cls, 
												state = torch.cat(state), 
												action = torch.cat(action))
		
		elif isinstance(state, TensorDict) and isinstance(action, torch.Tensor):
			return super(Transition,cls).__new__(cls, 
												state = state, 
												action = action)
		else:
			raise NotImplementedError(f"Not NotImplementedError for types: {type(state)} and {type(action)}")

	@property
	def _is_batch(self)->bool:
		return self.state.shape[0] > 1

	def __len__(self):
		return self.state.shape[0]

# class ReplayMemory(object):
#     def __init__(self, capacity:int):
#         self.memory = deque([], maxlen=capacity)
#         self.capacity = capacity

#     