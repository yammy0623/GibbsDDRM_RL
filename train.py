from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import A2C, DQN, PPO, SAC
from gymnasium import spaces
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F
import os
import gc
from main import parse_args_and_config
from runners.my_diffusion import MyDiffusion

th.set_printoptions(sci_mode=False)
WANDB=False

warnings.filterwarnings("ignore")
register(
    id='final-ours',
    entry_point='envs:DiffusionEnv',
)
register(
    id='final-baseline',
    entry_point='envs:BaselineDiffusionEnv',
)

def make_env(my_config):
    def _init():
        config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "threshold": my_config["threshold"],
            "DM": my_config["DM"],
        }
        if my_config["agent2"] is not None:
            config["agent2"] = my_config["agent2"]
        
        if my_config["model_mode"] == "baseline":
            print('Baseline training mode ...')
            return gym.make('final-baseline', **config)
        else:
            print('2-agent training mode ...')
            config["agent1"] = my_config["agent1"]
            return gym.make('final-ours', **config)
    return _init

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, use_scale_shift_norm: bool = True):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['image'].shape[0]
        self.use_scale_shift_norm = use_scale_shift_norm
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Normalize features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space['image'].sample()[None]).float()
            ).shape[1]

        self.fc = nn.Linear(1, 32)
        self.fc2 = nn.Linear(1, 32)
        self.fc_merge = nn.Linear(64, 32)
        self.embedding_output = nn.Linear(32, features_dim * 2)
        self.out_norm = nn.Linear(n_flatten, features_dim)  # Normalizing layer
        self.out_rest = nn.Sequential(
            nn.Linear(features_dim, features_dim),  # Further processing layer
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        img_features = self.cnn(observations['image'].float())
        value_features = None

        if 'value' in observations:
            value_features = F.relu(self.fc(observations['value'].float()))

        if 'remain' in observations:
            remain_features = F.relu(self.fc2(observations['remain'].float()))
            if value_features is not None:
                value_features = self.fc_merge(th.cat([value_features, remain_features], dim=1))
            else:
                value_features = remain_features  

        if self.use_scale_shift_norm:
            if value_features is not None:
                emb_out = self.embedding_output(value_features)
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = self.out_norm(img_features) * (1 + scale) + shift
            else:
                h = self.out_norm(img_features) 
            h = self.out_rest(h)
        else:
            if value_features is not None:
                h = self.out_rest(self.out_norm(img_features + value_features))
            else:
                h = self.out_rest(self.out_norm(img_features))

        return h


def eval(env, model, eval_episode_num, num_steps):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_reward = 0
    avg_reward_t = [0 for _ in range(num_steps)]
    avg_t = [0 for _ in range(num_steps)]
    avg_ssim = 0
    avg_psnr = 0
    pivot_ssim = 0
    pivot_psnr = 0
    ddim_ssim = 0
    ddim_psnr = 0
    ddnm_ssim = 0
    ddnm_psnr = 0
    avg_start_t = 0
    with th.no_grad():
        for seed in range(eval_episode_num):
            done = False
            # Set seed using SB3 API
            # env.seed(seed)
            obs, info = env.reset(seed=seed)

            now_t = 0
            # Interact with env using SB3 API
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                avg_reward_t[now_t] += info['reward']
                now_t += 1
            
            avg_reward += info['reward']
            avg_ssim   += info['ssim']
            avg_psnr += info['psnr']
            pivot_ssim += info['pivot_ssim']
            pivot_psnr += info['pivot_psnr']
            ddim_ssim += info['ddim_ssim']
            ddim_psnr += info['ddim_psnr']
            ddnm_ssim += info['ddnm_ssim']
            ddnm_psnr += info['ddnm_psnr']
            # avg_start_t += info['time_step_sequence'][0]
            for i in range(num_steps):
                avg_t[i] += info['time_step_sequence'][i]

    avg_reward /= eval_episode_num
    avg_ssim /= eval_episode_num
    avg_psnr /= eval_episode_num
    pivot_ssim /= eval_episode_num
    pivot_psnr /= eval_episode_num
    ddim_ssim /= eval_episode_num
    ddim_psnr /= eval_episode_num
    ddnm_ssim /= eval_episode_num
    ddnm_psnr /= eval_episode_num
    avg_start_t /= eval_episode_num
    for i in range(num_steps):
        avg_reward_t[i] = (avg_reward_t[i] / eval_episode_num)
        avg_t[i] = avg_t[i] / eval_episode_num
    
    return avg_reward, avg_ssim, avg_psnr, pivot_ssim, pivot_psnr, ddim_ssim, ddim_psnr, ddnm_ssim, ddnm_psnr, info['time_step_sequence'], info['action_sequence'], info['threshold'], avg_reward_t, avg_t

def train(eval_env, model, config, epoch_num, second_stage=False, num_steps=5):
    """Train agent using SB3 algorithm and my_config"""
    current_best_psnr = 0
    current_best_ssim = 0
    print("total training epochs: ", int(epoch_num))
    for epoch in range(int(epoch_num)):

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
            progress_bar=True,
        )

        th.cuda.empty_cache()  # Clear GPU cache
        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_reward, avg_ssim, avg_psnr, pivot_ssim, pivot_psnr, ddim_ssim, ddim_psnr, ddnm_ssim, ddnm_psnr, time_step_sequence, action_sequence, threshold, avg_reward_t, avg_t = eval(eval_env, model, config["eval_episode_num"], num_steps)

        print("---------------")

        ### Save best model
        if (current_best_psnr + current_best_ssim) < (avg_psnr + avg_ssim) and (current_best_psnr < avg_psnr):# and epoch > 10:
        # if current_best_psnr < avg_psnr and current_best_ssim < avg_ssim:# and epoch > 10:
            print("Saving Model !!!")
            current_best_psnr = avg_psnr
            current_best_ssim = avg_ssim
            save_path = config["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if second_stage:
                model.save(f"{save_path}/best_2")
            else:
                model.save(f"{save_path}/best")


        print("Threshold:", threshold)
        print("Avg_reward:  ", avg_reward)
        print("Avg_reward_t:  ", avg_reward_t)
        print("Avg_t:  ", avg_t)
        print("Avg_ssim:    ", avg_ssim)
        print("Avg_psnr:    ", avg_psnr)
        print("Current_best_ssim:", current_best_ssim)
        print("Current_best_psnr:", current_best_psnr)
        print("Pivot_ssim:  ", pivot_ssim)
        print("Pivot_psnr:  ", pivot_psnr)
        print("DDIM_ssim:   ", ddim_ssim)
        print("DDIM_psnr:   ", ddim_psnr)
        print("DDNM_ssim:   ", ddnm_ssim)
        print("DDNM_psnr:   ", ddnm_psnr)
        print("Time_step_sequence:", time_step_sequence)
        print("Action_sequence:", action_sequence)
        print()
        if WANDB:
            wandb.log(
                {
                    "avg_reward": avg_reward,
                    "avg_ssim": avg_ssim,
                    "avg_psnr": avg_psnr,
                    "pivot_ssim": pivot_ssim,
                    "pivot_psnr": pivot_psnr,
                    "ddim_ssim": ddim_ssim,
                    "ddim_psnr": ddim_psnr,
                    "ddnm_ssim": ddnm_ssim,
                    "ddnm_psnr": ddnm_psnr,
                    "start_t": avg_t[0],
                }
        )


def main():
    # Initialze DDNM
    args, config = parse_args_and_config()
    runner = MyDiffusion(args, config)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    my_config = {
        "algorithm": A2C,
        "target_steps": args.target_steps,
        "threshold": 0.9,
        "num_train_envs": 1, # 16

        "epoch_num": 200,
        "first_stage_epoch_num": 50,
        "policy_network": "MultiInputPolicy",
        "timesteps_per_epoch": 100,
        "eval_episode_num": 1, # 16
        "learning_rate": 1e-4, 
        "policy_kwargs": policy_kwargs,

        "max_steps": 100,
        "task": args.deg,
        "model_mode": "baseline" if args.baseline else "2_agents",
        "finetune": args.finetune,

    }
    
    my_config['run_id'] = f'{my_config["task"]}_{args.path_y}_{my_config["model_mode"]}_A2C_env_{my_config["num_train_envs"]}_steps_{my_config["target_steps"]}'
    if args.baseline == False:
        if args.second_stage:
            my_config['run_id'] += '_S2'
        else:
            my_config['run_id'] += '_S1'

    my_config['save_path'] = f'model/{my_config["task"]}_{args.path_y}_{my_config["model_mode"]}_A2C_{my_config["target_steps"]}'
    if WANDB:
        run = wandb.init(
            project="final",
            config=my_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            id=my_config["run_id"],
        )
    
    config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "threshold": my_config["threshold"],
            "DM": runner,
            "model_mode": my_config["model_mode"],
        }
    if args.baseline == False:
        config["agent1"] = None

    # Create training environment 
    num_train_envs = my_config['num_train_envs']
    
    if not args.second_stage:
        ### First stage training
        if args.baseline:
            epoch_num = my_config['epoch_num'] 
        else: 
            epoch_num = my_config["first_stage_epoch_num"]

        if args.finetune:
            epoch_num = epoch_num *0.1
            print("finetune: training with epoch_num = ", epoch_num)
        else:
            print("training with epoch_num = ", epoch_num)

        ### First stage finetuning stage (need to load agent 2)
        if args.finetune:
            agent2 = my_config["algorithm"].load(f"{my_config['save_path']}/best_2")
            config['agent2'] = agent2
            train_env = DummyVecEnv([make_env(config) for _ in range(num_train_envs)])
            
        else:
            config['agent2'] = None
            train_env = DummyVecEnv([make_env(config) for _ in range(num_train_envs)])
        

        model = my_config["algorithm"](
                my_config["policy_network"], 
                train_env, 
                verbose=2,
                tensorboard_log=os.path.join("tensorboard_log", my_config["run_id"]),
                learning_rate=my_config["learning_rate"],
                policy_kwargs=my_config["policy_kwargs"],
            )
        del config["model_mode"], config["agent2"]
        eval_env = gym.make('final-baseline', **config) if args.baseline else gym.make('final-ours', **config)
        
        train(eval_env, model, my_config, epoch_num = epoch_num, num_steps = args.target_steps)
        
    else:
        ### Second stage training
        epoch_num = my_config['epoch_num']
        if args.finetune:
            epoch_num = epoch_num *0.1
            print("finetune: training with epoch_num = ", epoch_num)
        else:
            print("training with epoch_num = ", epoch_num)

        print("Loaded model from: ", f"{my_config['save_path']}/best")
        agent1 = my_config["algorithm"].load(f"{my_config['save_path']}/best")
        config['agent1'] = agent1
        config['agent2'] = None
        config["model_mode"] = my_config["model_mode"]
        train_env = DummyVecEnv([make_env(config) for _ in range(num_train_envs)])
        del config["model_mode"], config['agent2']
        eval_env = gym.make('final-ours', **config)
        model2 = my_config["algorithm"](
            my_config["policy_network"], 
            train_env, 
            verbose=2,
            tensorboard_log=os.path.join("tensorboard_log", my_config["run_id"]),
            learning_rate=my_config["learning_rate"],
            policy_kwargs=my_config["policy_kwargs"],
        )
        train(eval_env, model2, my_config, epoch_num = epoch_num, second_stage=True, num_steps = args.target_steps)

if __name__ == '__main__':
    main()
