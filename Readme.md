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
                        batch_data, weights <- memory_buffer.sample() # with priority memory
                        
```
- In Double DQN, $a'\rightarrow Q_{net}$ and $next\_q\_value\rightarrow target\_Q\_net$
```
                        td_errors, huber_loss <- compute from batch_data
```

- Update priority with **TD** errors and $\epsilon$
```
                        memory_buffer.update_priority(param = td_errors + some `epsilon`)
                        
                        # gradient accumualation
                        huber_loss <- huber_loss*td_errors*weights
                        huber_loss.backward()
                    
                    # update DQN after gradient accumualation
                    optimizer.step() with learning rate lr
                    optimizer.zero_grad()
                # end train_model

```
- Soft update target network by copying weights from main network with hyperparameter $\tau$
```
            if update_target:
                target_net <- soft update with `tau` <- DQN_net


```





q_sampled_action <- Q_net forward with observations (from memory)


best_next_action = torch.argmax(double_q, -1) which double_q <- Q_net(next_state)

target_q <- target_net(next_state)
best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)



q_update = reward + self.gamma * best_next_q_value * (1 - done)
td_error = q_sampled_action - q_update





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
