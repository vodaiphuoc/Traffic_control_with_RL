from collections import namedtuple, deque
import random
import torch
from tensordict.tensordict import TensorDict

if __name__ == "__main__":
    device = torch.device('cpu')
    trans = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    mem = deque([], maxlen= 5)

    obj = trans(1,2,3,4)

    print({k:v for k,v in obj._asdict().items()})