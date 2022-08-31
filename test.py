import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from environment import create_atari_env
from model import ActorCriticModel


def test(rank, params, shared_model):
    """ Test the model by playing a game and showing it on the screen """
    torch.manual_seed(params.seed + rank)
    
    # Create environment
    env = create_atari_env(params.env_name, video=True)
    env.seed(params.seed + rank)
    
    # Create model
    model = ActorCriticModel(env.observation_space.shape[0], env.action_space.n)
    model.eval()
    
    # Init state
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    

    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0
    
    # Let the agent play the game
    while True:
        episode_length += 1
        
        # Update hidden state
        if done:
            model.load_state_dict(shared_model.state_dict())
            lstm_c = Variable(torch.zeros(1, 256), volatile=True)
            lstm_h = Variable(torch.zeros(1, 256), volatile=True)
        else:
            lstm_c = Variable(lstm_c.data, volatile=True)
            lstm_h = Variable(lstm_h.data, volatile=True)
            
        # Forward pass of the state through the model
        _, action_values, (lstm_h, lstm_c) = model((Variable(state.unsqueeze(0), volatile=True), (lstm_h, lstm_c)))
        
        # Determine the action to play
        action_probabilities = F.softmax(action_values)
        action = action_probabilities.max(1)[1].data.numpy()
        
        # Perform the action
        state, reward, done, _ = env.step(action[0, 0])
        
        # Keep track of the reward
        reward_sum += reward
        
        # If the game is over log the result and reset for the next game
        if done:
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)
            
        # Prepare next state
        state = torch.from_numpy(state)
