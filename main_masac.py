import argparse
import torch
import os, random
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from replaybuffer_ma import ReplayBuffer

from rl_algorithms.masac_change.masac import MASAC
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# refer code for paper: Actor Attention Critic for MultiAgent Reinforcement Learning
# paper url: https://arxiv.org/abs/1810.02912
# code url: https://github.com/shariqiqbal2810/MAAC
USE_CUDA = False  # torch.cuda.is_available()
def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_environment(config):
        # check environment to close the environment that might not have been closed before
        try:
            env.close()
        except:
            pass

        # make environment and call channels to communicate with unity exe
        engine_channel = EngineConfigurationChannel()
        params_channel = EnvironmentParametersChannel()
        
        exe_path = config.proj_path + config.env_path + config.env_name + '/UnityEnvironment.exe'
        env = UnityEnvironment(file_name= exe_path, worker_id= config.worker_id, base_port= config.base_port, side_channels=[engine_channel, params_channel])

        if config.run_type == "train":
            engine_channel.set_configuration_parameters(time_scale=20, target_frame_rate= -1, capture_frame_rate= 60, quality_level= 5)

        # reset environment to collect the state and action size info
        env.reset()
        
        behavior_names = list(env.behavior_specs)
        num_agents = len(behavior_names) # 2
        actions_dim = list()
        states_dim = list()
        for i in range(num_agents):
            spec = env.behavior_specs[behavior_names[i]]
            actions_dim.append(spec.action_spec.continuous_size)
            states_dim.append(spec.observation_shapes[0][0])

        # modify the force which could be added into the joint of each joint [-1, 1]
        # follow instructions of using EngineConfigurationChannel and EnvironmentParametersChannel:
        # https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-API.md

        params_channel.set_float_parameter("force_magnitude", 100.0)
        params_channel.set_float_parameter("terminate_steps", config.episode_length)

        params_channel.set_float_parameter("target_distance", 0.02)
        params_channel.set_float_parameter("target_angle", 5.0)

        params_channel.set_float_parameter("use_hand_angle", 1.0)
        params_channel.set_float_parameter("use_comfort_pose", 0.0)
        params_channel.set_float_parameter("use_hand_distance", 1.0)
        params_channel.set_float_parameter("use_palm_touch_board", 1.0)
        params_channel.set_float_parameter("use_ball_balance", 1.0)
        params_channel.set_float_parameter("use_board", 1.0)

        # params_channel.set_float_parameter("add_dynamic_board", 1.0)
        # params_channel.set_float_parameter("add_lifting_board", 1.0)

        params_channel.set_float_parameter("hand_distance_ratio", 15.0) #3
        params_channel.set_float_parameter("hand_angle_ratio", 2.0)
        params_channel.set_float_parameter("palm_touch_board_ratio", 2.0)
        params_channel.set_float_parameter("ball_balance_ratio", 2.0)
        params_channel.set_float_parameter("board_ratio", 1.5)
        params_channel.set_float_parameter("comfort_pose_ratio", 1.5)
        
        return env, states_dim, actions_dim, num_agents, behavior_names

def run_train(config):
    model_dir = Path(config.proj_path + config.model_path + config.env_name)
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    #env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    env, states_dim, actions_dim, num_agents, behavior_names = make_environment(config)

    model = MASAC.init_from_env(env,states_dim, actions_dim,
                                       tau=config.tau,
                                       lr=config.lr,
                                       gamma=config.gamma,
                                       hidden_dim=config.hidden_dim)

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents, states_dim, actions_dim)
    t = 0
    count_explore_step = 0 
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        env.reset()
        temp = []
        for i in range(num_agents):       
            decision_steps, terminal_steps = env.get_steps(behavior_names[i])
            temp.append(decision_steps.obs[0])
        obs = np.array(temp)

        model.prep_rollouts(device='cpu')

        DONE = False
        step_i = 0
        while not DONE and step_i<config.episode_length:
            step_i += 1
            count_explore_step += 1

            # to start using training actions until finishing the explore_step
            if count_explore_step > config.explore_step:
                # rearrange observations to be per agent, and convert to torch Variable
                torch_obs = [Variable(torch.Tensor(obs[i]),requires_grad=False)
                            for i in range(model.nagents)]
                # get actions as torch Variables
                torch_agent_actions = model.step(torch_obs, evaluate=False)
                # convert actions to numpy arrays
                actions = [ac.data.numpy() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                #actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                # print("actions train: ", type(actions), actions)
            else:
                actions = [np.array([[random.uniform(-1,1) for _ in range(actions_dim[0])]], dtype=('float32')) for _ in actions_dim]
                # print("actions random: ", type(actions), actions)
            #return np.array([[random.uniform(-1,1) for _ in range(len(action[0]))]])

            # print("train actions: ", actions[0][0], actions[1][0])

            for i in range(num_agents):
                # print("current actions: ", ep_i, "---",step_i,"---", actions[i]  )
                action_tuple = ActionTuple()
                action_tuple.add_continuous(actions[i])
                env.set_actions(behavior_names[i], action_tuple)

            env.step()

            # if ep_i > 364 and step_i>26:
            #     print("its more 350 episodes now")
            next_obs = [None]*num_agents
            rewards = [None]*num_agents
            dones = [None]*num_agents
            for i in range(num_agents):
                decision_steps, terminal_steps = env.get_steps(behavior_names[i])
                if len(decision_steps) > 0:
                    dones[i] = False
                    next_obs[i] = decision_steps.obs[0]
                    rewards[i] = decision_steps.reward * config.reward_scale
                if len(terminal_steps) > 0:
                    dones[i] = True
                    next_obs[i] = terminal_steps.obs[0]
                    rewards[i] = terminal_steps.reward * config.reward_scale # add reward_scale to make sure the entropy term and Q term close to each other

            replay_buffer.push(obs, actions, rewards, next_obs, dones)
            obs = next_obs

            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')

                for _ in range(config.num_updates):
                    for a_i in range(model.nagents):
                        # sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                        sample = replay_buffer.sample_N_1(config.batch_size, to_gpu=config.use_gpu)
                        model.update(sample, a_i, logger=logger)

                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
            
            # to terminate for the current episode if the terminate state reach
            for done in dones:
                if done:
                    DONE = True

        # ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        #ep_rews = replay_buffer.get_average_step_rewards(step_i)
        ep_rews = replay_buffer.get_average_episode_rewards(ep_i, step_i)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)
        if ep_i%50==0 and ep_i!=0:
            print("Episodes %i of %i" % (ep_i + 1, config.n_episodes), ", steps ", step_i, ", agent 0 mean_rewards: %.5f "% ep_rews[0], ", agent 1 mean_rewards: %.5f "% ep_rews[1])
        

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

def run_test(config):
    model_path = Path(config.proj_path + config.model_path + config.env_name)/ ('run%i' % config.run_num)
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   (config.incremental*config.save_interval+1))
    else:
        model_path = model_path / 'model.pt'

    model = MASAC.init_from_save(model_path)
    env, _, _, num_agents, behavior_names = make_environment(config)

    model.prep_rollouts(device='cpu')

    for ep_i in range(config.n_episodes_test):
        env.reset()

        temp = []
        for i in range(num_agents):       
            decision_steps, terminal_steps = env.get_steps(behavior_names[i])
            temp.append(decision_steps.obs[0])
        obs = np.array(temp)

        DONE = False
        step_i = 0
        total_rewards=[0]*num_agents
        while not DONE and step_i<config.episode_length_test:
            step_i += 1
            torch_obs = [Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(model.nagents)]

            # get actions as torch Variables
            torch_actions = model.step(torch_obs, evaluate=True) # evaluate=True, make the posture more stable, evaluate=False, make the posture more vibrate
            # convert actions to numpy arrays
            actions = [ac.data.numpy() for ac in torch_actions]

            # print("test actions: ", actions[0][0], actions[1][0])
            for i in range(num_agents):
                action_tuple = ActionTuple()
                action_tuple.add_continuous(actions[i])
                env.set_actions(behavior_names[i], action_tuple)
            env.step()

            next_obs = [None]*num_agents
            dones = [None]*num_agents
            rewards = [None]*num_agents
            for i in range(num_agents):
                decision_steps, terminal_steps = env.get_steps(behavior_names[i])
                if len(decision_steps) > 0:
                    dones[i] = False
                    next_obs[i] = decision_steps.obs[0]
                    rewards[i] = decision_steps.reward
                if len(terminal_steps) > 0:
                    dones[i] = True
                    next_obs[i] = terminal_steps.obs[0]
                    rewards[i] = terminal_steps.reward
                total_rewards[i] += rewards[i]
            
            obs = next_obs

            for done in dones:
                if done:
                    DONE = True
            
        print("Episodes %i of %i" % (ep_i + 1, config.n_episodes), ", steps ", step_i, "average rewards: %.5f" % (total_rewards[0][0]/step_i), ", %.5f" %(total_rewards[1][0]/step_i))

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--proj_path', default= 'D:/BoxSync/projects2020/marl_mlagent_torch/RLcode/')
    parser.add_argument('--env_path', default= 'unity_exes/')
    parser.add_argument('--model_path', default= 'model_results/masac_change/')
    parser.add_argument('--env_name', default= 'woodybalance40', help= 'name of the environment to train with')

    parser.add_argument('--base_port', default= 5008, help= 'the base port use to communicate with unity environment')
    parser.add_argument('--worker_id', default=7, type=int,help='use to separate the socket communication channel between python and Unity environment' )
    parser.add_argument('--run_type', type = str, default= "train", help= "choose to train or test")

    parser.add_argument("--model_name", default='', help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=11, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int) # 1e4
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--n_episodes_test", default=20, type=int)
    parser.add_argument("--episode_length", default=3000, type=int)
    parser.add_argument("--explore_step", default=10000, type=int, help="explore certain steps randomly before using the training action ouput")
    parser.add_argument("--episode_length_test", default=50, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int, help="Number of updates per update cycle")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training") # 256, 128
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int) # 256, 128
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float) # 3e-4 , 1e-3
    parser.add_argument("--tau", default=0.005, type=float) # 0.005, 0.001
    parser.add_argument("--gamma", default=0.99, type=float) # 0.96
    parser.add_argument("--reward_scale", default=1., type=float) # 100
    parser.add_argument("--use_gpu", action='store_true')

    ### parameters for evaluation
    parser.add_argument("--run_num", default=2, type=int)
    parser.add_argument("--incremental", default=10, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")

    config = parser.parse_args()

    if config.run_type == 'train':
        run_train(config)
    
    if config.run_type == 'test':
        run_test(config)
