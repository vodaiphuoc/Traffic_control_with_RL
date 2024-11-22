#### Description
Traffic control with deep Q learning and SUMO environment and Pytorch model
- Policy network takes inputs include information of SUMO map and a binary image which
represents vehicle positions return from the environment.
- Implement relay memory with NamedTuple, deque in Python and TensorDict (Pytorch)

#### Pseudo Workflow:
```
1:  for _ in range(number_of_episodes): 
2:      env.reset()

3:      for step in steps():
4:          action <- greedy-epsilon policy # based on random_action and DQN
5:          state, action, next_state, reward = env.step(action)
6:          memory_buffer.push(new_transition) 

7:          if memory.is_full():
8:              # trigger train_model() function
                def train_model():
9:                  for i in range(num_epochs):
10:                     batch_data, weights <- memory_buffer.sample() # with priority memory
                        
```

- In Double DQN, $`a'\leftarrow Q_{net}`$ and $`next\_q\_value\leftarrow target\_Q\_net`$
```
11:                     q_value <- Q_net(state_batch)
12:                     next_batch_action <- Q_net(next_state_batch).argmax()
13:                     next_q_value <- target_Q_net(next_state_batch).gather(-1, next_batch_action)
14:                     td_errors <- abs(next_q_value, q_value)
15:                     huber_loss <- next_q_value, q_value, reward
```

- Update priority with **TD** errors and $\epsilon$
```
16:                     memory_buffer.update_priority(param = td_errors + some `epsilon`)
                        
                        # gradient accumualation
17:                     huber_loss <- huber_loss*td_errors*weights
                        huber_loss.backward()
                    
                    # update DQN after gradient accumualation
18:                 optimizer.step() with learning rate lr
                    optimizer.zero_grad()
                # end train_model

```
- Soft update target network by copying weights from main network with hyperparameter $\tau$
```
            if update_target:
19:             target_net <- soft update with `tau` <- DQN_net

```

#### New workflow

##### Offline training stage
- **Input**: 
    - behaviour policy list: List[$`\pi_{\beta\_i}`$] = [random, pre-trained, greedy-epsilon policies]
    - empty memory_relay $D_{off}$ with capacity $C$ and contains $N$ buckets

1. action $\leftarrow$ behaviour policy
2. state, action, next_state, reward $\leftarrow$ env.step(action)
3. Store new_transition in offline memory_relay $D_{off}$
4. **If** $D_{off}$ is full:
5. &ensp; &ensp; Training Double deep Q network $Q_{net}$

##### Deploy stage: Inference + online training
- **Input**:
    - trained DDQN network $Q_{net}$, 
    - filled offline memory_relay $D_{off}$
    - empty memory_relay $D_{onl}$ with specification like $D_{off}$

1. action $\leftarrow Q_{net}\leftarrow$ state $\leftarrow$ env.reset()
2. state, action, next_state, reward $\leftarrow$ env.step(action)
3. Store new_transition $D_{onl}[buck_{j,j\in N}]$
4. **If** $buck_{j}$ waitl is full **then**:
5. &ensp; &ensp; **TD** errors $\leftarrow$ inference $\leftarrow buck_{j}$ data
6. &ensp; &ensp; Set corresponding priorities (via **TD** errors)
7. &ensp; &ensp; **for** $j=1$ to $N$ **do**
7. &ensp; &ensp; &ensp; &ensp; Flattening _state_ and _next_ _state_
8. &ensp; &ensp; &ensp; &ensp; Calculate [MMD](https://pytorch.org/ignite/generated/ignite.metrics.MaximumMeanDiscrepancy.html)$(buck_{j}, D_{off}[buck_{j}])$
9. &ensp; &ensp; **end for**
10. &ensp; &ensp; $MMD(buck_{j}, D_{off})\leftarrow {1/N}\sum_{j=1}^N MMD(buck_{j}, D_{off}[buck_{j}])$
11. **If** $j\geq2$ **and** $MMD(buck_{j}, D_{off}) > MMD(buck_{j-1}, D_{off})*ratio$ **then**: # Trigger training $Q_{net}$
12. &ensp; &ensp; Sampling data from $D_{onl}[buck_{0:j}]$ + $D_{off}$ with priorities
13. **else**:
14. &ensp; &ensp; back to **step 1)**.



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






other refs:
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
