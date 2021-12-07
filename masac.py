from math import trunc
import torch
import torch.nn.functional as F

from utils import soft_update
from rl_algorithms.masac_crpo.sac_crpo import SACAgent

MSELoss = torch.nn.MSELoss()

class MASAC(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = 2
        self.agents = [SACAgent(lr=lr,
                                hidden_dim=hidden_dim,
                                **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    def step(self, observations, evaluate=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, evaluate=evaluate) for a, obs in zip(self.agents,
                                                                 observations)]

    def update (self, sample, agent_i, constrain_reach=[False, False], parallel=False, logger=None):

        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        # rews shape = (2, 256, 3)
        # sample batch from memory
        ########### update the constrain value network
        curr_agent.constrain_critic_optimizer.zero_grad()

        out = [pi.sample(nobs) for pi, nobs in zip(self.policies, next_obs)]
        all_trgt_acs = [out[0][0], out[1][0]]
        next_state_log_pi = out[agent_i][1]
        
        trgt_q_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        q1_next_target, q2_next_target = curr_agent.constrain_target_critic(trgt_q_in)
        min_q_next_target = torch.min(q1_next_target,  q2_next_target) - curr_agent.alpha*next_state_log_pi
        y_target = rews[agent_i][:,0].view(-1,1) + (1-dones[agent_i].view(-1,1))*self.gamma*min_q_next_target
        # two q-functions to mitigate positive bias in the policy improvement step
        q_in = torch.cat((*obs, *acs), dim=1)
        q1, q2 = curr_agent.constrain_critic(q_in)
        q1_loss = F.mse_loss(q1, y_target)
        q2_loss = F.mse_loss(q2, y_target)
        constrain_q_loss = q1_loss + q2_loss

        constrain_q_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(curr_agent.constrain_critic.parameters(), 0.5)
        curr_agent.constrain_critic_optimizer.step()

        ########### update value network
        curr_agent.critic_optimizer.zero_grad()

        out = [pi.sample(nobs) for pi, nobs in zip(self.policies, next_obs)]
        all_trgt_acs = [out[0][0], out[1][0]]
        next_state_log_pi = out[agent_i][1]
        
        trgt_q_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        q1_next_target, q2_next_target = curr_agent.target_critic(trgt_q_in)
        min_q_next_target = torch.min(q1_next_target,  q2_next_target) - curr_agent.alpha*next_state_log_pi
        y_target = rews[agent_i][:,1].view(-1,1) + (1-dones[agent_i].view(-1,1))*self.gamma*min_q_next_target
        # two q-functions to mitigate positive bias in the policy improvement step
        q_in = torch.cat((*obs, *acs), dim=1)
        q1, q2 = curr_agent.critic(q_in)
        q1_loss = F.mse_loss(q1, y_target)
        q2_loss = F.mse_loss(q2, y_target)
        q_loss = q1_loss + q2_loss

        q_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        ########## update policy network
        curr_agent.policy_optimizer.zero_grad()
        out1 = [pi.sample(nobs) for pi, nobs in zip(self.policies, obs)]
        all_actions = [out1[0][0], out1[1][0]]
        state_log_pi = out1[agent_i][1]
        q_p_in = torch.cat((*obs, *all_actions), dim=1)
        # add  the constrain check here to select the right critic function to use
        if constrain_reach[agent_i]:
            q1_value, q2_value = curr_agent.critic(q_p_in)
        else:
            q1_value, q2_value = curr_agent.constrain_critic(q_p_in)

        min_q_pi = torch.min(q1_value, q2_value)

        policy_loss = (curr_agent.alpha*state_log_pi - min_q_pi).mean()

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        ########## update entropy, tuning entropy parameter alpha
        curr_agent.alpha_optimizer.zero_grad()
        alpha_loss = -(curr_agent.log_alpha*(state_log_pi + curr_agent.target_entropy).detach()).mean()
        alpha_loss.backward()
        curr_agent.alpha_optimizer.step()

        curr_agent.alpha = curr_agent.log_alpha.exp()
        alpha_tlogs = curr_agent.alpha.clone() # for tensorboardx logs


        # print("self.niter--------: ", len(self.niter))
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'q_loss': q_loss,
                                'pol_loss': policy_loss,
                                "log_pi": state_log_pi.mean().item(),
                                'constrain_q_loss': constrain_q_loss,
                                'alpha': alpha_tlogs}, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            # soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            # a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def nsave(self, filename, obs_norm):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                        'agent_params': [a.get_params() for a in self.agents]}
        save_total = [obs_norm.mean, obs_norm.std, save_dict]
        torch.save(save_total, filename)

    @classmethod
    def init_from_env(cls, env, states, actions,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for acsp, obsp in zip(actions, states):
            num_in_pol = obsp
            num_out_pol = acsp
 
            num_in_critic = 0
            for oobsp in states:
                num_in_critic += oobsp
            for oacsp in actions:
                num_in_critic += oacsp  


            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'agent_init_params': agent_init_params}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

    @classmethod
    def init_from_nsave(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        # save_dict = torch.load(filename)
        o_mean, o_std, save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance, o_mean, o_std
