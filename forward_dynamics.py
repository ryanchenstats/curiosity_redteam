from typing import List, Callable, Union
import os
import random
import numpy as np
from multiprocessing import Pool

from embedders.embedders import Embedder

from torch import nn
import torch 

class ForwardDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, embedder: Embedder, device='cpu'):
        super(ForwardDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.embedder = embedder
        self.to(device=device)
        self.optimizer = torch.optim.NAdam(self.model.parameters(), lr = 0.0001)
        self.criteria = nn.MSELoss(reduction='none')
        self.max_loss = -1.0

    def forward(self, state, action):
        state_embedding = self.embedder(state)
        action_embedding = self.embedder(action)
        x = torch.cat([state_embedding, action_embedding], dim=1)
        next_state = self.model(x)
        return next_state
    
    def train(self, state, action, next_state):
        self.optimizer.zero_grad()
        next_state_embedding = self.embedder(next_state)
        pred_next_state_embedding = self.forward(state, action)
        loss = self.criteria(pred_next_state_embedding, next_state_embedding)
        per_state_loss = torch.mean(loss, dim=-1)
        
        batch_max = torch.max(per_state_loss).item()
        if batch_max > self.max_loss:
            self.max_loss = batch_max
        
        per_state_loss = per_state_loss / self.max_loss
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        return per_state_loss
    
    def eval_bonus(self, state, action, next_state):
        with torch.no_grad():
            next_state_embedding = self.embedder(next_state)
            pred_next_state_embedding = self.forward(state, action)
            loss = self.criteria(pred_next_state_embedding, next_state_embedding)
            per_state_loss = torch.mean(loss, dim=-1)
            per_state_loss = per_state_loss / self.max_loss
            
        return per_state_loss
