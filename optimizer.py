import math
import torch
import torch.optim as optim

class SharedAdam(optim.Adam):
    
    def __init__(self, params, lr = 1e-3, betas =(0.9,0.999), eps=1e-8, weight_decay=0):
        """
        Initialize the Adam Optimizer
        
        @param params: iterable of parameters to optimize
        @param lr: learning rate
        @param betas: coefficients used for computing running averages of gadient and its square
        @param eps: term added to the denominator to improve numerical stability
        @param weight_decay: weight decay
        """
        
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        
        #All parameters to optimize
        for group in self.param_groups:
            #tensor p of weights to optimize
            for p in group['params']:
                state = self.state[p]
                #tensor([0])
                state['step'] = torch.zeros(1)
                #exponential moving average of the gradient
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                #exponential moving average of the squared of the gradient
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
        
    def share_memory(self):
        """
        Sharing the memory
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                #moving underlying storage to shared memory
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    def step(self):
        """
        Performs a single optimization step.

        """
        super(SharedAdam, self).step()
        