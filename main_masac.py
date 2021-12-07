import argparse
import yaml
import torch
import os, random
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from replaybuffer_ma import ReplayBuffer
from normalizer import normalizer

from rl_algorithms.masac_crpo.masac_crpo import MASAC
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
        
        exe_path = config['proj_path'] + config['env_path'] + config['env_name'] + '/UnityEnvironment.exe'
        env = UnityEnvironment(file_name= exe_path, worker_id= config['worker_id'], base_port= config['base_port'], side_channels=[engine_channel, params_channel])

        if config['run_type'] == "train":
            engine_channel.set_configuration_parameters(time_scale=20, target_frame_rate= -1, capture_frame_rate= 60, quality_level= 5)
        if config['run_type']  == 'test':
            engine_channel.set_configuration_parameters(time_scale=1, target_frame_rate= -1, capture_frame_rate= 60, quality_level= 5)
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

        params_channel.set_float_parameter("force_magnitude", config['env']['force_magnitude'])
        params_channel.set_float_parameter("terminate_steps", config['train']['steps_train'])
        params_channel.set_float_parameter("remove_support_episodes", config['env']['remove_support_episodes'])

        params_channel.set_float_parameter("target_distance", config['env']['target_distance'])
        params_channel.set_float_parameter("target_angle", config['env']['target_angle'])
        params_channel.set_float_parameter("target_maintain_time", config['env']['target_maintain_time'])

        params_channel.set_float_parameter("ball_balance_ratio", config['env']['ball_balance_ratio']) #5
        params_channel.set_float_parameter("board_ratio", config['env']['board_ratio']) #5
        params_channel.set_float_parameter("comfort_pose_ratio", config['env']['comfort_pose_ratio']) #1.5
        params_channel.set_float_parameter("trimatch_ratio", config['env']['trimatch_ratio'])

        # for camera
        params_channel.set_float_parameter("cam_rot_speed", config['env']['cam_rot_speed'])
        params_channel.set_float_parameter("cam_look_distance", config['env']['cam_look_distance'])

        # for observation
        params_channel.set_float_parameter("observe_index", config['env']['observe_index'])
        params_channel.set_float_parameter("action_index", config['env']['action_index'])
        params_channel.set_float_parameter("use_support", config['env']['use_support'])

        # for choosing a reward function to use from a different tried functions
        params_channel.set_float_parameter("comfort_dense_reward_index", config['env']['comfort_dense_reward_index'])
        params_channel.set_float_parameter("comfort_sparse_reward_index", config['env']['comfort_sparse_reward_index'])

        params_channel.set_float_parameter("board_dense_reward_index", config['env']['board_dense_reward_index'])
        params_channel.set_float_parameter("board_sparse_reward_index", config['env']['board_sparse_reward_index'])

        params_channel.set_float_parameter("ball_balance_dense_reward_index", config['env']['ball_balance_dense_reward_index'])
        params_channel.set_float_parameter("ball_balance_sparse_reward_index", config['env']['ball_balance_sparse_reward_index'])

        params_channel.set_float_parameter("trimatch_dense_reward_index", config['env']['trimatch_dense_reward_index'])
        params_channel.set_float_parameter("trimatch_sparse_reward_index", config['env']['trimatch_sparse_reward_index'])

        params_channel.set_float_parameter("print_pythonparams", config['env']['print_pythonparams'])
        params_channel.set_float_parameter("print_rewardlog", config['env']['print_rewardlog'])

        params_channel.set_float_parameter("time_penalty_reward", config['env']["time_penalty_reward"])
        
        return env, states_dim, actions_dim, num_agents, behavior_names

def run_train(config):
    model_dir = Path(config['proj_path'] + config['model_path'] + config['env_name'])
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

    torch.manual_seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    #env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    env, states_dim, actions_dim, num_agents, behavior_names = make_environment(config)

    if(config['train']['load_init_policy']):
        policy_path=Path(config['proj_path'] + config['model_path'])/config['train']['load_init_policy']
        model,o_mean,o_std = MASAC.init_from_nsave(policy_path)
        config['train']['explore_step'] = 0
        print("load from save")
        obs_norm = normalizer(size=states_dim[0], default_clip_range=config['train']['clip_range'], mean=o_mean, std=o_std)
    else:
        print("start from env")
        model = MASAC.init_from_env(env,states_dim, actions_dim,
                                        tau=config['train']['tau'],
                                        lr=config['train']['lr'],
                                        gamma=config['train']['gamma'],
                                        hidden_dim=config['train']['hidden_dim'])
        obs_norm = normalizer(size=states_dim[0], default_clip_range=config['train']['clip_range'])

    replay_buffer = ReplayBuffer(config['train']['buffer_length'], model.nagents, states_dim, actions_dim)
    t = 0
    count_explore_step = 0 
    sum_episode_rewards=[0]*num_agents
    for ep_i in range(0, config['train']['episodes_train'], config['train']['n_rollout_threads']):
        env.reset()
        obs = []
        for i in range(num_agents):       
            decision_steps, terminal_steps = env.get_steps(behavior_names[i])
            obs.append(decision_steps.obs[0])
        # obs = np.array(temp)

        model.prep_rollouts(device='cpu')

        DONE = False
        step_i = 0
        episode_rewards=[0]*num_agents
        while not DONE and step_i<config['train']['steps_train']:
            step_i += 1
            count_explore_step += 1

            # to start using training actions until finishing the explore_step
            if count_explore_step > config['train']['explore_step']:
                # rearrange observations to be per agent, and convert to torch Variable
                n_obs = obs_norm.normalize(obs)
                torch_obs = [Variable(torch.Tensor(n_obs[i]),requires_grad=False)
                            for i in range(model.nagents)]
                # get actions as torch Variables
                torch_agent_actions = model.step(torch_obs, evaluate=False)
                # convert actions to numpy arrays
                actions = [ac.data.numpy() for ac in torch_agent_actions]
            else:
                actions = [np.array([[random.uniform(-1,1) for _ in range(actions_dim[0])]], dtype=('float32')) for _ in actions_dim]

            for i in range(num_agents):
                action_tuple = ActionTuple()
                action_tuple.add_continuous(actions[i])
                env.set_actions(behavior_names[i], action_tuple)

            env.step()

            # if step_i>298:
            #     print("its more 350 episodes now")
            next_obs = [None]*num_agents
            rewards = [None]*num_agents
            dones = [None]*num_agents
            for i in range(num_agents):
                decision_steps, terminal_steps = env.get_steps(behavior_names[i])
                if len(decision_steps) > 0:
                    dones[i] = False
                    next_obs[i] = decision_steps.obs[0]
                    rewards[i] = decision_steps.reward * config['train']['reward_scale']
                if len(terminal_steps) > 0:
                    dones[i] = True
                    next_obs[i] = terminal_steps.obs[0]
                    rewards[i] = terminal_steps.reward * config['train']['reward_scale'] # add reward_scale to make sure the entropy term and Q term close to each other
                episode_rewards[i] += rewards[i]
            replay_buffer.push(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            # episode_rewards =[episode_rewards[i]+rewards[i] for i in range(num_agents)]

            t += config['train']['n_rollout_threads']
            if (len(replay_buffer) >= config['train']['batch_size'] and (t % config['train']['steps_per_update']) < config['train']['n_rollout_threads']):
                if config['train']['use_gpu']:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')

                for _ in range(config['train']['num_updates']): # 4
                    for a_i in range(model.nagents):
                        # sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                        sample = replay_buffer.sample(config['train']['batch_size'], to_gpu=config['train']['use_gpu'])
                        sample = norm_update(sample, obs_norm)
                        model.update(sample, a_i, logger=logger)
                        # q p update

                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
            
            # to terminate for the current episode if the terminate state reach
            for done in dones:
                if done:
                    DONE = True
                    # print("done: ", DONE)

            #  calculate monte carlo, a
            #  J = discount* r_height + ....,    Q(s,a)=expection(r_height), Q(S,A) = EXP(r_target)
        # ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        #ep_rews = replay_buffer.get_average_step_rewards(step_i)
        sum_episode_rewards =[sum_episode_rewards[i] + episode_rewards[i] for i in range(num_agents)]
        ep_rews = [r/(ep_i+1) for r in sum_episode_rewards]
        #ep_rews = replay_buffer.get_average_episode_rewards(ep_i, step_i)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config['train']['steps_train'], ep_i)
        if ep_i%100==0 and ep_i!=0:
            print("Episodes %i of %i" % (ep_i + 1, config['train']['episodes_train']), ", steps ", step_i, ", agent 0 mean_rewards: %.5f "% ep_rews[0], ", agent 1 mean_rewards: %.5f "% ep_rews[1])
        
        if ep_i % config['train']['save_interval'] < config['train']['n_rollout_threads']:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.nsave(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)), obs_norm)
            model.nsave(run_dir / 'model.pt', obs_norm)

    model.nsave(run_dir / 'model.pt', obs_norm)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

def run_test(config):
    model_path=Path(config['proj_path'] + config['model_path'])/config['test']['test_model']
    if config['test']['test_model']:
        model, o_mean, o_std = MASAC.init_from_nsave(model_path)
    else:
        print("Please provide a model path to load.")
        return
    env, states_dim, _, num_agents, behavior_names = make_environment(config)

    obs_norm = normalizer(size=states_dim[0], default_clip_range=config['train']['clip_range'], mean=o_mean, std=o_std)

    model.prep_rollouts(device='cpu')

    for ep_i in range(config['test']['episodes_test']):
        env.reset()

        temp = []
        for i in range(num_agents):       
            decision_steps, terminal_steps = env.get_steps(behavior_names[i])
            temp.append(decision_steps.obs[0])
        obs = np.array(temp)

        DONE = False
        step_i = 0
        total_rewards=[0]*num_agents

        # print("episode start: -----------------")
        # print("obs: ", obs[0], " ------- ", obs[1])
        while not DONE and step_i<config['test']['steps_test']:
            step_i += 1
            n_obs = obs_norm.normalize(obs)
            torch_obs = [Variable(torch.Tensor(n_obs[i]), requires_grad=False) for i in range(model.nagents)]

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

            # print("obs: ", obs[0], " ------- ", obs[1])
            for done in dones:
                if done:
                    DONE = True
            
        print("Episodes %i of %i" % (ep_i + 1, config['test']['episodes_test']), ", steps ", step_i, "average rewards: %.5f" % (total_rewards[0][0]), ", %.5f" %(total_rewards[1][0]))

    env.close()

def norm_update(sample, obs_norm):
    # the input in torch.tensor type
    obs, acs, rews, next_obs, dones = sample
    
    obs = (torch.stack(obs)).numpy() # to convert a list of tensors into a tensor shape, [(256,49), (256,49)] --> (2,256,49), and then convert to numpy 
    next_obs = (torch.stack(next_obs)).numpy()

    # the normalizer function needs the input data to be in numpy type
    obs_norm.update(obs)
    obs_norm.recompute_stats()

    obs = obs_norm.normalize(obs)
    next_obs = obs_norm.normalize(next_obs)
    # convert back to torch tensor
    obs = torch.from_numpy(obs)
    next_obs = torch.from_numpy(next_obs)

    return obs, acs, rews, next_obs, dones

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='', help='the path of the config yaml file used for the current run')
    out = parser.parse_args()
    # for line by line debug purpose
    # out.config_path = 'configs/masac_change/train/s14_wb71_run1.yml'

    with open(out.config_path, 'r') as fp:
        try: 
            config = yaml.safe_load(fp)
        except:
            print("yaml file load failed!")

    if config['run_type'] == 'train':
        run_train(config)
    
    if config['run_type'] == 'test':
        run_test(config)
