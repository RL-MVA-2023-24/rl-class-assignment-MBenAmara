# %%writefile train.py
# DQN config
config = {'nb_actions': 4,
          'learning_rate': 0.00003,
          'gamma': 0.95,
          'buffer_size': 10000,
          'epsilon_min': 0.01,
          'epsilon_max': 0.95,
          'epsilon_decay_period': 50000,
          'epsilon_delay_decay': 200000,
          'batch_size': 16,
          'nb_neurons': 512,
          'n_action' : 4,
          'episodes' : 5000,
          'Qupdate': 500}


from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import os
import numpy as np
import torch
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import torch
import torch.nn as nn
import gymnasium as gym
import random
import numpy as np
from copy import deepcopy
import pandas as pd
import numpy as np
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons= config['nb_neurons']

class Model(nn.Module):
    def __init__(self, input_dim = state_dim,
                 hidden1_dim = nb_neurons,
                 hidden2_dim = nb_neurons,
                 output_dim = n_action) :

        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        # self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    # device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()



class ProjectAgent:
  def __init__(self) : 
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.gamma = config['gamma']
      self.batch_size = config['batch_size']
      self.nb_actions = config['nb_actions']
      self.memory = ReplayBuffer(config['buffer_size'], device)
      self.epsilon_max = config['epsilon_max']
      self.epsilon_min = config['epsilon_min']
      self.epsilon_stop = config['epsilon_decay_period']
      self.epsilon_delay = config['epsilon_delay_decay']
      self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
      self.model = Model().to(device)
      self.Tmodel = deepcopy(self.model).to(device)
      self.criterion = torch.nn.MSELoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
      self.episodes = config["episodes"]
      self.Qupdate = config["Qupdate"]


  def gradient_step(self):
    if len(self.memory) > self.batch_size:
        X, A, R, Y, D = self.memory.sample(self.batch_size)
        QYmax = self.Tmodel(Y).max(1)[0].detach()
        #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
        update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
        QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = self.criterion(QXA, update.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

  def act(self, obs) : 
        action = greedy_action(self.model, obs)
        return action
  

  def load(self) : 
    try :
        self.model.load_state_dict(torch.load("modelBest"))
        print("Loaded")
    except :
        self.model.load_state_dict(torch.load("modelBest",map_location=torch.device('cpu')))
        print("Loaded")

  def train(self, env):
        max_episode = self.episodes
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        acts = []
        best = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
                acts.append(action)

            # step
            next_state, reward, trunc, done, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if step % 5000 == 0 or  step == 1 :
                name = f"model"
                torch.save(self.model.state_dict(),name)
                pd.DataFrame(episode_return).to_csv("scores.csv")
                print("################################################################################################")
                print(pd.Series(acts).value_counts())
                print(f"Best score is {best}")
                acts = []
                print("########################################## Model Saved #########################################")

            if step % self.Qupdate == 0 : 
              self.Tmodel = deepcopy(self.model).to(device)
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", len Mem ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward/1e6),
                      ", steps ", step,
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                if episode_cum_reward > best : 
                    print("Best Saved")
                    best = episode_cum_reward
                    name = f"modelBest"
                    torch.save(self.model.state_dict(),name)

                episode_cum_reward = 0


            else:
                state = next_state



                
        return episode_return
  



#agent = ProjectAgent()
#agent.train(env)