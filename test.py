from src.environment.env import SumoEnvironment
import torch
# from src.agents.dqn_agent import ReplayMemory
from collections import namedtuple, deque
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


if __name__ == "__main__":
    device = torch.device('cpu')
    mem = ReplayMemory(capacity= 5)


    sumo_env = SumoEnvironment(
        map_config_path= "src\\nets\\2way-single-intersection\\map_config.json",
        net_file="src\\nets\\2way-single-intersection\single-intersection.net.xml",
        route_file="src\\nets\\2way-single-intersection\single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_traffic_light=True,
        use_gui=True,
        num_seconds=100000
    )

    state, info = sumo_env.reset()

    observation, reward, terminated, truncated, infor = sumo_env.step(0)
    # print("observation: ",observation)
    
    mem.push(state, action = torch.tensor([0]), next_state = observation, reward = 0.2)
    mem.push(state, action = torch.tensor([1]), next_state = observation, reward = 0.2)
    mem.push(state, action = torch.tensor([2]), next_state = observation, reward = 0.2)

    rand_batch = mem.sample(batch_size= 3)

    print(type(rand_batch))

    batch_state, batch_action, batch_next_state, batch_reward = mem.get_batch(rand_batch)

    print(batch_state)