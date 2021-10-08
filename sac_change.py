import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.distributions import Normal
from utils import hard_update, gumbel_softmax, onehot_from_logits, OUNoise
import numpy as np
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain = 1)
        torch.nn.init.constant_(m.bias, 0)

# def weights_init_(m):
#     if isinstance(m):
#         torch.nn.init.orthorgonal_(m.weight, 1.0)
#         torch.nn.init.constant_(m.bias, 1e-6)

class SACAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=256,
                 lr=0.01, alpha=0.2):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.alpha = alpha
        self.policy = GaussianPolicy(num_in_pol, num_out_pol,
                                 hidden_dim, hidden_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        # if automatic_entropy_tune:
        #self.target_entropy = -num_out_pol # torch.prod(torch.Tensor(num_out_pol).to(self.device)).item() # self.target_entropy = - action_dim
        self.target_entropy = - np.log(num_out_pol)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        #self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr = lr)

        self.critic = QNetwork(num_in_critic, 1,
                                 hidden_dim,hidden_dim)
        self.target_critic = QNetwork(num_in_critic, 1,
                                        hidden_dim, hidden_dim)
        hard_update(self.target_critic, self.critic)
        
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

    def step(self, obs, evaluate=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        if not evaluate:
            action, _, _ = self.policy.sample(obs)
        else:
            _, _, action = self.policy.sample(obs)

        return action #.detach().cpu().data.numpy()

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=256, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size, fc2_size):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.Q1_critic = nn.Sequential(
            nn.Linear(input_dim, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_dim)   
        )
        # Q2 architecture
        self.Q2_critic = nn.Sequential(
            nn.Linear(input_dim, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_dim)   
        )

        self.apply(weights_init_)

    def forward(self, xu):

        # xu = torch.cat((state, action), 1)

        q1 = self.Q1_critic(xu)
        q2 = self.Q2_critic(xu)

        return q1, q2


class GaussianPolicy(nn.Module):
    def __init__(self, state_size, action_size, fc1_size, fc2_size, action_space = None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(state_size, fc1_size)
        self.linear2 = nn.Linear(fc1_size, fc2_size)
        self.mean_linear = nn.Linear(fc2_size, action_size)
        self.log_std_linear = nn.Linear(fc2_size, action_size)

        self.apply(weights_init_)

        # action rescaling

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
          
            self.action_scale = torch.tensor(1.0) # torch.FloatTensor((action_space.high - action_space.low)/2.0)
            self.action_bias = torch.tensor(0.0) # torch.FloatTensor((action_space.high + action_space.low)/2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min = LOG_SIG_MIN, max = LOG_SIG_MAX)


        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std*N(0,1))
        y_t = torch.tanh(x_t)

        action = y_t*self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # enforcing action bound
        log_prob -= torch.log(self.action_scale*(1-y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim = True)
        mean = torch.tanh(mean)*self.action_scale + self.action_bias

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
