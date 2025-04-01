import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

# Mô hình DQN đơn giản
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 1, *shape))
        return int(torch.numel(o))
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Thêm channel dimension
        return self.fc(self.conv(x))

# Replay Memory để lưu trữ trải nghiệm
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, experience):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)
        
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
            
    def __len__(self):
        return len(self.memory)
