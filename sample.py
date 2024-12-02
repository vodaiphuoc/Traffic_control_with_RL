from src.memory.memories_v2 import Transition, ReplayMemory, Base_Transition, Bucket
from tensordict.tensordict import TensorDict
import torch
import numpy as np
import random
from collections import namedtuple
import time
import gc


# from pympler import asizeof

def memory_summary():
	# Only import Pympler when we need it. We don't want it to
	# affect our process if we never call memory_summary.
	from pympler import summary, muppy, tracker
	mem_summary = summary.summarize(muppy.get_objects())
	rows = summary.format_(mem_summary)
	print('\n'.join(rows))
	tr = tracker.SummaryTracker()
	tr.print_diff()


capacity = 10**6
memory = Bucket(bucket_size = capacity, device = torch.device('cpu'))

memory_summary()

for i in range(capacity):
	new_time = torch.from_numpy(np.random.random_sample((5,))).to(torch.float32)
	new_queue = torch.from_numpy(np.random.random_sample((3,))).to(torch.float32)
	density = torch.from_numpy(np.random.random_sample((8,))).to(torch.float32)
	memory.add(state = {
					'time': new_time,
					'queue': new_queue,
					'density': density
				},
				action = 3,
				next_state = {
					'time': new_time,
					'queue': new_queue,
					'density': density
				},
				reward = 0.9
				) 


memory = memory.clear()
time.sleep(5)

collected = gc.collect()
print("Garbage collector: collected",
          "%d objects." % collected)

memory_summary()

print(memory)