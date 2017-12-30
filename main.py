from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import my_optim

class Params():
    """
    Gathering all the parameters that are required to explore
    """
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'
        
# Main run

#1 thread per core
os.environ['OMP_NUM_THREADS'] = '1'
params = Params()
torch.manual_seed(params.seed)
env = create_atari_env(params.env_name)

#shared_model is the model shared by the different agents (different threads in different cores)
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space) 
shared_model.share_memory() 

optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
#store in shared memory
optimizer.share_memory() 

#Processes
processes = []
p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
p.start()
processes.append(p)

#Run all the other processes that will be trained by updating the shared model
for rank in range(0, params.num_processes):
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)
    
#Creating a pointer that will allow to kill all the threads when at least one of the threads,
#or main.py will be killed, allowing to stop the program safely
for p in processes:
    p.join()
