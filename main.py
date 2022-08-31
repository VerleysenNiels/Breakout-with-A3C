from __future__ import print_function

import os
from test import test

import torch
import torch.multiprocessing as mp

import shared_adam_optimizer
from environment import create_atari_env
from model import ActorCriticModel
from train import train


# Training parameters
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'


# MAIN
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Init the training parameters
    params = Params()
    
    # Set the seed and create the environmend
    torch.manual_seed(params.seed)
    env = create_atari_env(params.env_name)
    
    # Create the shared model
    shared_model = ActorCriticModel(env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()
    
    # Init the optimizer and link it to the shared model
    optimizer = shared_adam_optimizer.SharedAdam(shared_model.parameters(), lr=params.lr)
    optimizer.share_memory()
    
    # Set up a single process that continuously tests the latest model
    processes = []
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
    p.start()
    processes.append(p)
    
    # Setup multiple agents to train on the game
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
        p.start()
        processes.append(p)
        
    # Wait until all processes are finished
    for p in processes:
        p.join()
