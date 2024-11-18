from src.agent.dqn_agent import DQN
import torch
import random

from tensordict.tensordict import TensorDict
from typing import List, Dict, Literal, Union


class Double_Q_Agent(object):
	def __init__(self, 
				action_space:int, 
				device: torch.device, 
				hyper_params: Dict[str, Union[int,float]]
				)->None:

		self.Q_net = DQN(n_actions = action_space, 
                 	CNN_out_dim = 128, 
                 	other_future_dim = 21).to(device)

		self.target_Q_net = DQN(n_actions = action_space,
						CNN_out_dim = 128,
			            other_future_dim = 21).to(device)

		self.target_Q_net.load_state_dict(self.Q_net.state_dict())

		self.steps_done = 0
		self.device = device
		self.hyper_params = hyper_params

	def select_action(self,
                      state: TensorDict,
                      env: SumoEnvironment
                      )-> torch.Tensor:
		"""Sampling action from Env with epsilon greedy policy"""
        sample = random.random()
        eps_threshold = self.hyper_params['EPS_END'] + (self.hyper_params['EPS_START'] - self.hyper_params['EPS_END'])* \
            math.exp(-1. * self.steps_done / self.hyper_params['EPS_DECAY'])
        
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = state.to(self.device)
                return self.Q_net(**state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], 
                                device=self.device, 
                                dtype=torch.long
            )

    def _compute_loss_TD_error(self):
    	

	def soft_update(self)->None:
		"""Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        policy_net_state_dict = self.Q_net.state_dict()
        target_net_state_dict = self.target_Q_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.hyper_params['TAU'] + \
            									target_net_state_dict[key]*(1-self.hyper_params['TAU'])
        
        self.target_Q_net.load_state_dict(target_net_state_dict)
        return None