from typing import Dict, Tuple, List, Union, Literal
from tensordict.tensordict import TensorDict
import numpy as np
import torch
from collections import namedtuple

# super class
Base_Transition = namedtuple('Base_Transition',['state','action'])

class Transition(Base_Transition):
	"""Class represents single transition or a batch of transition"""
	def __new__(cls,
				state: Union[TensorDict, Tuple[TensorDict]],
				action: Union[torch.Tensor, Tuple[torch.Tensor]],
				device: torch.device = torch.device('cpu')
				)->Base_Transition:
		if isinstance(state, tuple) and isinstance(action, tuple):
			return super(Transition,cls).__new__(cls, 
												state = torch.cat(state).to(device), 
												action = torch.cat(action).to(device))
		
		elif isinstance(state, TensorDict) and isinstance(action, torch.Tensor):
			return super(Transition,cls).__new__(cls,
												state = state.to(device), 
												action = action.to(device))
		else:
			raise NotImplementedError(f"Not NotImplementedError for types: {type(state)} and {type(action)}")

	@property
	def is_batch(self)->bool:
		return self.state.shape[0] > 1

	def __len__(self):
		return self.state.shape[0]

class ReplayMemory(Transition):
	"""
	Buffer relay memory which manipulate Transition
	ref: https://pytorch.org/tensordict/stable/tutorials/tensordict_preallocation.html
	"""
	def __new__(cls, 
				capacity:int, 
				device: torch.device = torch.device('cuda'),
				priority_sampling: bool = False,
				)->Transition:
		return super(ReplayMemory,cls).__new__(cls,
												state = TensorDict({}, batch_size = capacity),
												action = torch.empty(size = [capacity], dtype = torch.int8),
												device = device
												)
	def __init__(self,
				capacity:int,
				device: torch.device = torch.device('cuda'),
				priority_sampling: bool = False,
				)->None:
		self.current_index = -1
		self.capacity = capacity

	@property
	def is_full(self)->bool:
		return self.current_index+1 == self.capacity

	# def _random_sampling():



	def push(self, transition: Transition, index:int = None):
		
		

		self.current_index += 1

		self.state[self.current_index] = transition.state[0]
		self.action[self.current_index] = transition.action

