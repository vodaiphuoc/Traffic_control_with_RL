from typing import Dict, Tuple, List, Union, Literal
from tensordict.tensordict import TensorDict
import numpy as np
import torch
from collections import namedtuple

# super class
Base_Transition = namedtuple('Base_Transition',['state','action','next_state','reward'])

class Transition(Base_Transition):
	"""Class represents single transition or a batch of transition"""
	def __new__(cls,
				state: Union[TensorDict, Tuple[TensorDict]],
				action: Union[torch.Tensor, Tuple[torch.Tensor]],
				next_state: Union[TensorDict, Tuple[TensorDict]] = None,
				reward: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
				device: torch.device = torch.device('cpu')
				)->Base_Transition:
		if isinstance(state, tuple) and isinstance(action, tuple):
			return super(Transition,cls).__new__(cls, 
												state = torch.cat(state).to(device), 
												action = torch.cat(action).to(device),
												next_state = torch.cat(next_state).to(device),
												reward = torch.cat(reward).to(device))
		
		elif isinstance(state, TensorDict) and isinstance(action, torch.Tensor):
			return super(Transition,cls).__new__(cls,
												state = state.to(device), 
												action = action.to(device),
												next_state = next_state.to(device),
												reward = reward.to(device))
		else:
			raise NotImplementedError(f"Not NotImplementedError for types: {type(state)} and {type(action)}")

	@property
	def is_batch(self)->bool:
		return self.state.shape[0] > 1

	def __len__(self):
		return self.state.shape[0]


class Bucket(object):
	def __init__(self, bucket_size:int, device: torch.device):
		self.bucket_size = bucket_size
		self.device = device
		self.transitions = Transition(state = TensorDict({}, batch_size = bucket_size),
									action = torch.empty(size = [bucket_size], dtype = torch.int8),
									next_state = TensorDict({}, batch_size = bucket_size),
									reward = torch.empty(size = [bucket_size], dtype = torch.float32),
									device = device)
		self.priority_min = None
		self.priority_sum = None
		self.next_idx = 0

	def add(self, 
			state: Dict[str, np.ndarray],
			action: int,
			next_state: Dict[str, np.ndarray],
			reward: float
		)->None:
		# init tree only has data add into first index
		if self.next_idx == 0:
			self.priority_min = [float('inf') for _ in range(2*self.bucket_size)]
			self.priority_sum = [0 for _ in range(2*self.bucket_size)]

		# set data into transitions
		self.transitions.state[self.next_idx] = state
		self.transitions.action[self.next_idx] = action
		self.transitions.next_state[self.next_idx] = next_state
		self.transitions.reward[self.next_idx] = reward

		# incease next index
		self.next_idx += 1

	@property
	def is_bucket_full(self)->bool:
		return self.next_idx == self.bucket_size

	def compute_priorities(self):
		"""
		From all data stored in bucket, inferece with DDQN
		to get TD errors, then 
		"""
		pass		

	def clear(self):
		del self.transitions
		return self.__init__(self.bucket_size, self.device)


class ReplayMemory(object):
	"""
	Buffer relay memory which manipulate Transition
	ref: https://pytorch.org/tensordict/stable/tutorials/tensordict_preallocation.html
	"""
	def __init__(self, 
				number_buckets:int,
				bucket_size:int, 
				device: torch.device = torch.device('cuda')
				)->None:
		self.bucket_size = bucket_size
		self.buckets = {
			ith: {
				'transition':Transition(state = TensorDict({}, batch_size = bucket_size),
										action = torch.empty(size = [bucket_size], dtype = torch.int8),
										device = device),
				'index': -1
				}
			for ith in range(number_buckets)
		}
		self.current_bucket_index = 0

	@property
	def is_bucket_full(self, bucket_index:int)->bool:
		return self.buckets[bucket_index]['index'] + 1 == self.bucket_size

	def push(self, transition: Transition, index:int = None):
		
		

		self.current_index += 1

		self.state[self.current_index] = transition.state[0]
		self.action[self.current_index] = transition.action

