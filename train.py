import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    """
    Make sure the models share the same gradient
    
    @param model
    @param shared_model
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        
        if shared_param.grad is not None:
            return
        
        shared_param._grad = param.grad
        

def train(rank, params, shared_model, optimizer):
    """
    Train the AI
    
    @param rank: rank to shift the seed in order to asynchronize the agent
    @param params: all of the parameters of environment
    @param shared_model: What the agent run its exploration
    @param optimizer: 
    """
    
    torch.manual_seed(params.seed + rank)
    
    #optimized environment
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    #numpy array of size 1*42*42
    state = env.reset()
    state = torch.from_numpy(state)
    
    #game finished -> done is True
    done = True
    
    episode_length = 0 
    
    while True:
        episode_length+=1
        #load share model to exploration
        model.load_state_dict(shared_model.state_dict())
        
        # Cell state and Hidden state of the LSTM are reinitialized to zero
        # if the game is done or first iteration
        if done:
            cx = Variable(torch.zeros(1, 256)) 
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
            
        #list of values (V(S))
        #list of log probabilities
        #list of rewards
        #list of entropies
        values = []
        log_probs = []
        rewards = []
        entropies =[]
        
        #exploration
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            
            action = prob.multinomial().data
            
            log_prob = log_prob.gather(1, Variable(action))
            log_probs.append(log_prob)
            #playing the selected action, reaching the new state, and getting the new reward
            state, reward, done, _ = env.step(action.numpy())
            #If the agent last too long, then done 
            done = (done or episode_length >= params.max_episode_length)
            
            reward = max(min(reward, 1), -1) 

            if done:
                #restart environment
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            rewards.append(reward)

            if done:
                break

        #Intializing the cumulative reward
        R = torch.zeros(1,1)

        if not done:
            #Initialize the cumulative reward with the value of the last shared state
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data 
        
        values.append(Variable(R))

        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        
        #Initializing the Generalized Advantage Estimation to 0
        gae = torch.zeros(1, 1) 

        for i in reversed(range(len(rewards))):
            #R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
            R = params.gamma * R + rewards[i] 
            #Advantage_i = Q_i - V(state_i) = R - value[i]
            advantage = R - values[i] 
            #Computing the value loss
            value_loss = value_loss + 0.5 * advantage.pow(2)
            #Computing the temporal difference
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data 
            # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
            gae = gae * params.gamma * params.tau + TD 
            #Computing the policy loss
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] 
        
        #optimizer
        optimizer.zero_grad() 
        #policy loss is more important than value loss
        (policy_loss + 0.5 * value_loss).backward() 
        #clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) 
        ensure_shared_grads(model, shared_model)
        optimizer.step()