import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 

def normalized_columns_intializer(weights, std = 1.0):
    """
    Intialize and setting the standard deviation of the tensor of weights

    @param wegihts: weights needed to intialize
    @param std: standard deviation
    @return normalized weights
    """
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))

    return out

def weights_init(m):
    """
    Initialize the weights of the NN for an optimal learning

    @param m: Neural Network object(Convolution or Full connection)
    """
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        #dim1 * dim2 * dim3
        fan_in = np.prod(weight_shape[1:4])
        #dim0 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
       
        w_bound = np.sqrt(6. / (fan_in + fan_out) )
        # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound)
        #fill the tensor of bias with 0
        m.bias.data.fill_(0)
        
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())

        fan_in = weight_shape[1]
        fan_out = weight_shape[0]

        w_bound = np.sqrt(6. / (fan_in + fan_out))

        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    


class ActorCritic(torch.nn.Module):
    """
    A3C AI Brain
    
    """
    def __init__(self, num_inputs, action_space):
        """
        Initialize the ActorCritic
        
        @param num_inputs: # of inputs
        @param action_space: action space
        """
        
        super(ActorCritic, self).__init__()
        
        #Four Convolution layers
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        
        #Long-short term memory: learn the temporal property of the input
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        #number of q values
        num_outputs = action_space.n
        
        #output = V(s)
        self.critic_linear = nn.Linear(256, 1)
        #output =  Q(S,A)
        self.actor_linear = nn.Linear(256, num_outputs)
        
        #initialize weights
        self.apply(weights_init)
        #setting the standard deviation
        self.actor_linear.weight.data = normalized_columns_intializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill(0)
        
        self.critic_linear.weight.data = normalized_columns_intializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill(0)
        
        self.lstm.bias_ih.data.fill(0)
        self.lstm.bias_hh.data.fill(0)
        
        self.train()
    
    def forward(self, inputs):
        """
        NN Forward Propagate function
        
        @param inputs: input images
        @return output of the critic, output of the actor, and the new hidden & cell states
        """
        #Separete into hidden state and cell states
        inputs, (hx, cx) = inputs
        
        #Propagate Convolutional layers
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        
        #Flattening into 1D vector
        x = x.view(-1, 32 * 3 * 3)
        #LSTM takes x and the old hidden & cell states
        #ouputs the new hidden & cell states
        hx, cx = self.lstm(x, (hx, cx))
        #Get hidden state
        x = hx 
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)