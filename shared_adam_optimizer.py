import math
import torch
import torch.optim as optim

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, (beta1, beta2), epsilon, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1) # We are at step 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_() # Init momentum to 0
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_() # Init average squared update to 0
                
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    
    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve data for current paramns
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Update the state
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Update the averages
                exp_avg.mul_(beta1).add_(1-beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1-beta2, grad, grad)
                
                # Compute the denominator for the update
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Compute the bias correction for the learning rate
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Update the weights
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss