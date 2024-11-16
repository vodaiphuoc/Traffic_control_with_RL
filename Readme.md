#### Descripttion \\
Traffic control with deep Q learning and SUMO environment and Pytorch model \\
- Policy network takes inputs include information of SUMO map and a binary image which \\
represents vehicle positions return from the environment.
- Implement relay memory with NamedTuple, deque in Python and TensorDict (Pytorch) \\


#### Workflow:
1)  for _ in range(number_of_episodes): \\
2)      action <- greedy-epsilon policy (based on random_action and DQN) \\
3)      state, action, next_state, reward = env.step(action) \\
4)      memory_buffer.push(new_transition) \\
5)      if memory.is_full(): \\
            # trigger train model \\
            def train_function(): \\
6)              for i in range(num_epochs): \\
7)                  batch_data, weights <- memory_buffer.sample() # with priority memory \\
7)                  td_errors, huber_loss <- compute from batch_data \\
8)                  memory.update_priority(param = td_errors + some_epsilon) \\
9)                  loss = loss*td_errors*weights/accumulation_steps \\
10)                 loss.backward() \\
11)                 if is_accumulation_step: \\
12)                     optimizer.step()     \\
13)                     optimizer.zero_grad()\\




- https://arxiv.org/pdf/2312.07795v1
- https://github.com/XingshuaiHuang/DTLight/tree/main

- https://arxiv.org/pdf/2303.10828v2
- https://github.com/LiangZhang1996/DataLight/tree/main
- make offline datasets:
    - https://github.com/LiangZhang1996/AttentionLight/tree/master


- https://arxiv.org/pdf/2310.05723

- https://traffic-signal-control.github.io/


- https://openreview.net/pdf?id=Q32U7dzWXpc
- https://github.com/sfujim/TD3_BC/tree/main

- https://www.sciopen.com/article/10.26599/AIR.2023.9150020


#### References: 
@misc{AlegreSUMORL,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}

@misc{
    author = {Adam Paszke, Mark Towers},
    title = {{Reinforcement Learning (DQN) Tutorial}},
    publisher = {PyTorch},
    howpublished = {\url{https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html}},
}
