import torch
import torch.nn.functional as F
from torch.autograd import Variable

from environment import create_atari_env
from model import ActorCriticModel

# Used to make sure that multiple models share gradients
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
    
def train(rank, params, shared_model, optimizer):
    """ Training function
    
        Rank is used to desynchronize different agents through the seed
        Params contains the parameters for this training process
        shared_model
        optimizer is the optimizer used in training
    
    """    
    
    torch.manual_seed(params.seed + rank) # Seed
    
    environment = create_atari_env(params.env_name) # Create environment
    environment.seed(params.seed + rank) # Environment seed
    
    model = ActorCriticModel(environment.observation_space.shape[0], environment.action_space.n) # Neural network
    
    # Prepare input state
    state = environment.reset()
    state = torch.from_numpy(state)
    done = True
    
    episode_length = 0
    
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        
        if done:
            # Game is done, so we need to reset the hidden state
            lstm_h = Variable(torch.zeros(1, 256))   
            lstm_c = Variable(torch.zeros(1, 256))    
        else:
            # Game is still going on
            lstm_h = Variable(lstm_h.data)   
            lstm_c = Variable(lstm_c.data)  
        
        # Lists to be filled in during exploration
        values = []   # Critic outputs
        log_probs = [] 
        rewards = []
        entropies = []
        
        # Exploration for a given number of steps or until the game is over
        for step in range(params.num_steps):
            # Forward pass of the state through the model
            value, action_values, (lstm_h, lstm_c) = model((Variable(state.unsqueeze(0)), (lstm_h, lstm_c)))
            
            # Softmax layer to select the action to take
            action_probabilities = F.softmax(action_values)
            log_action_probabilities = F.log_softmax(action_values)
            
            # Select action
            action = action_probabilities.multinomial().data
            
            log_action_probabilities = log_action_probabilities.gather(1, Variable(action))
            
            # Compute entropy
            entropy = -(log_action_probabilities * action_probabilities).sum(1)
            
            # Update lists for this step
            values.append(value)
            log_probs.append(log_action_probabilities)
            entropies.append(entropy)
            
            # Perform the action in the environment
            state, reward, done = environment.step(action.numpy())
            
            # Prevent the agent from getting stuck in the game by automatically resetting when the game is going on for too long
            done = (done or episode_length >= params.max_episode_length)
            
            # Make sure the reward is between -1 and 1
            reward = max(min(reward, 1), -1)
            rewards.append(reward)
            
            # If the game is over or there is an automatic reset -> reset the game and leave the loop
            if done:
                episode_length = 0
                state = torch.from_numpy(environment.reset())
                break
            
            # Continue playing the game
            torch.from_numpy(state)
        
        # Prepare to train the model on what has been experienced    
        R = torch.zeros(1, 1)
        if not done:
            # Get the estimated value of the last state in case the game was not finished yet
            value, _, _ = model(Variable(state.unsqueeze(0)), (lstm_h, lstm_c))
            R = value.data
        
        # Value of last state is either 0 if the game is over or the predicted value if the game is still ongoing
        values.append(Variable(R))
        
        # Init variables for training
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)        
        
        for i in reversed(range(len(rewards))):
            # Compute cumulative reward
            R = params.gamma * R + rewards[i]
            
            # Compute value loss with advantage
            # Advantage is cumulative reward - value of this state
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            # Compute policy loss
            temporal_difference = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + temporal_difference
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
        
        # Compute gradients    
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        
        # Backprop
        optimizer.step()
            
