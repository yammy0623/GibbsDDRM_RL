import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch
import os
from PIL import Image
from gymnasium.spaces import Box, Dict
import os
import random
from datasets import get_dataset, data_transform, inverse_data_transform
from skimage.metrics import structural_similarity
import gc
from tqdm import tqdm

class DiffusionEnv(gym.Env):
    def __init__(self, target_steps=10, max_steps=100, threshold=0.8, DM=None, agent1=None, agent2=None):
        super(DiffusionEnv, self).__init__()
        self.DM = DM
        self.agent1 = agent1
        self.agent2 = agent2
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000//target_steps)][::-1]
        # Threshold for the sparse reward
        self.final_threshold = threshold
        # adjust: False -> First subtask, True -> Second subtask
        self.adjust = True if agent1 is not None else False
        
        self.sample_size = 256
        # Maximum number of steps  (Baseline)
        self.max_steps = max_steps 
        # Count the number of steps
        self.current_step_num = 0 
        # Define the action and observation space
        if self.adjust: # Subtask 2
            self.action_space = gym.spaces.Box(low=-5, high=5)
            self.observation_space = Dict({
                "image": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
                # "image2": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32), # This is asked by TA but not working
                "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16),
                "remain": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
            })
        else: # Subtask 1
            self.action_space = spaces.Discrete(100) # Discrete action space
            # self.action_space = gym.spaces.Box(low=0, high=1) # Continuous action space
            self.observation_space = Dict({
                "image": Box(low=0, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
                # "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
            })

        # Initialize the random seed
        self.seed(232)
        self.reset()
        # print("Training data size:", len(self.DM.dataset))
        ### Generate target PSNR and SSIM

        
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
    
    def reset(self, seed=None, dataset=None):
        if seed is not None:
            self.seed(seed)
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []
        self.data_idx = random.randint(0, len(self.DM.dataset)-1)
        # self.x_orig, self.classes = self.DM.dataset[self.data_idx]
        # self.x_orig = self.x_orig.unsqueeze(0)

        data_iter = iter(self.DM.data_loader)
        if self.DM.config.data.dataset == "GoPro":
            self.y, self.x_orig = next(data_iter)
            self.y = data_transform(self.DM.config, self.y)
            self.x, _, self.x_orig, self.A_inv_y = self.DM.preprocess(self.x_orig, self.data_idx)
            self.y = self.y.to(self.x_orig.device)
        else:
            self.x_orig, self.classes = next(data_iter)
            self.x, self.y, self.x_orig, self.A_inv_y = self.DM.preprocess(self.x_orig, self.data_idx)
       
        ddim_x = self.x.clone()
        ddim_x0_t = self.A_inv_y.clone()
        self.x0_t = self.A_inv_y.clone()
        # Save DDIM performance
        with torch.no_grad():
            for i in range(self.target_steps):
                # print(i)
                ddim_t = (torch.ones(self.x.shape[0]) * self.uniform_steps[i])
                # ddim_t = torch.tensor(self.uniform_steps[i])
                if i != 0:
                    ddim_x, kernel = self.DM.get_noisy_x(self.x, ddim_t, ddim_x0_t, self.y, self.ddim_et)
                # else:
                #     self.ddim_x = self.DM.get_noisy_x(ddim_t, self.ddim_x0_t, initial=True)
                # print("ddim_x", ddim_x.shape)
                ddim_x0_t, self.ddim_et = self.DM.single_step_gibbsddrm(ddim_x, ddim_t)
                prev_t = ddim_t

        orig = inverse_data_transform(self.DM.config, self.x_orig[0]).to(self.DM.device)
        ddim_x = inverse_data_transform(self.DM.config, ddim_x0_t[0]).to(self.DM.device)
        ddim_mse = torch.mean((ddim_x - orig) ** 2)
        self.ddim_psnr = 10 * torch.log10(1 / ddim_mse).item()
        self.ddim_ssim = structural_similarity(ddim_x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        
        self.ddnm_psnr = 1.0
        self.ddnm_ssim = 1.0


        
        observation = {
            "image": self.x0_t[0].cpu(),  
            # "value": np.array([999]),
        }
        
        self.cnt = 0
        if self.adjust: # Second subtask
            with torch.no_grad():
                # Run subtask 1 to get the initial t
                action, _state = self.agent1.predict(observation, deterministic=True)
                print("action:", action)
                start_t = 10 * (1+action) - 1 # Discrete action space
                # start_t = 999 * (action)# + 1) / 2 # Continuous action space
                t = torch.ones(self.x.shape[0])*torch.tensor(int(max(0, min(start_t, 999))))
                self.previous_t = t
                prev_t = self.previous_t
                self.interval = int(t / (self.target_steps - 1)) 
                self.uniform_interval = self.interval

                
                # Save subtask1 performance
                ddim_x = self.x.clone()
                ddim_x0_t = self.A_inv_y.clone()
                with torch.no_grad():
                    for i in range(self.target_steps):
                        ddim_t = t - int(t / (self.target_steps - 1)) * i
                        # print('pivot_t:',i, ddim_t)
                        if i != 0:
                            ddim_x, kernel = self.DM.get_noisy_x(self.x, ddim_t, ddim_x0_t, self.y, ddim_et)
                        # else:
                        #     ddim_x, kernel = self.DM.get_noisy_x(self.x, ddim_t, ddim_x0_t, self.y, ddim_et, initial=True)
                        ddim_x0_t, ddim_et = self.DM.single_step_gibbsddrm(ddim_x, ddim_t)
                    self.cnt += 1
                
                ddim_x = inverse_data_transform(self.DM.config, ddim_x0_t[0]).to(self.DM.device)
                ddim_mse = torch.mean((ddim_x - orig) ** 2)
                self.pivot_psnr = 10 * torch.log10(1 / ddim_mse).item()
                self.pivot_ssim = structural_similarity(ddim_x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
                observation = {
                    "image": self.x0_t[0].cpu(), 
                    "value": np.array([999]),
                    "remain": np.array([self.target_steps]),
                }
        self.cnt = 0
        del ddim_x, ddim_x0_t, ddim_mse, orig
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU cache
        # images = (self.GT_image / 2 + 0.5).clamp(0, 1)
        # images = images.cpu().permute(0, 2, 3, 1).numpy()[0]
        # images = Image.fromarray((images * 255).round().astype("uint8"))
        # filename = os.path.join('img', f"GT_{self.current_step_num}.png")
        # images.save(filename)
        

        return observation, {}
    
    def step(self, action):
        truncate = True if self.current_step_num >= self.max_steps else False
        # Denoise current image at time t
        with torch.no_grad():
            ### RL step
            
            if self.adjust == False: # First subtask
                initial_t = torch.tensor(500)
                start_t = 10 * (1+action) - 1 # Discrete action space
                # start_t = 999 * (action)# + 1) / 2 # Continuous action space
                t = torch.ones(self.x.shape[0])*torch.tensor(int(max(0, min(start_t, 999))))
                # print('t:', t)
                # self.old_interval = initial_t // (self.target_steps - 1)
                self.interval = int(t / (self.target_steps - 1)) 
                if self.cnt != 0:
                    self.x, kernel = self.DM.get_noisy_x(self.x, t, self.x0_t, self.y, self.et, initial=True)
                # self.pivot_x, kernel = self.DM.get_noisy_x(self.x, initial_t, self.x0_t, initial=True)
                self.action_sequence.append(action.item())


                
            else: # Second subtask
                initial_t = self.previous_t - self.interval if self.current_step_num != 0 else self.previous_t
                # t = initial_t - self.uniform_interval * action
                t = initial_t - self.interval * action
                thres = 999 if self.current_step_num == 0 else self.time_step_sequence[-1]
                t = torch.ones(self.x.shape[0])*torch.tensor(int(max(0, min(t, thres))))
                # self.old_interval = self.interval
                self.interval = int(t / (self.target_steps - self.current_step_num - 1)) if (self.target_steps - self.current_step_num - 1) != 0 else self.interval
                if self.cnt != 0:
                    self.x, kernel = self.DM.get_noisy_x(self.x, t, self.x0_t, self.y, self.et) if self.current_step_num != 0 else self.DM.get_noisy_x(self.x, t, self.x0_t, self.y, self.et, initial=True)
                # self.pivot_x = self.DM.get_noisy_x(initial_t, self.x0_t, self.et)
                self.action_sequence.append(action.item())
            
            self.previous_t = t
            
            self.x0_t, self.et = self.DM.single_step_gibbsddrm(self.x, t)
            # self.pivot_x0_t, _,  self.pivot_et = self.DM.single_step_ddnm(self.pivot_x, self.y, initial_t, self.classes)
            self.time_step_sequence.append(t.item())

            # Run the remaining steps with uniform sampling to get x_0|t for reward calculation
            self.uniform_x0_t = self.x0_t.clone()
            self.uniform_et = self.et.clone()


            isFromDegraded = True
            ### inference the remaining steps by agent2 in subtask 1
            self.time_step_sequence2 = []
            if self.agent2:
                print("inference by agent2")
                self.agent2_x0_t = self.x0_t.clone()
                self.agent2_et = self.et.clone()

                # start from degraded image
                if isFromDegraded:
                    self.agent2_x0_t = self.A_inv_y.clone()
                    observation2 = {
                        "image": self.agent2_x0_t[0].cpu(),  
                        "value": np.array([999]),
                        "remain": np.array([self.target_steps]),
                    }
                    
                    # starting t is from agent 1 prediction
                    previous_t2 = self.previous_t.clone()
                    interval2 = self.interval
                    
                    # agent 2 needs target step - 1 steps to finish the denoising 
                    # (still need to modify the first step from agent1) 
                    for agent2_step_num in range(self.target_steps):
                        # print("remaining steps:", self.target_steps - agent2_step_num -1)
                        action2, _state = self.agent2.predict(observation2, deterministic=True)
                        
                        if agent2_step_num != 0 :
                            initial_t2 = previous_t2 - interval2 
                        else:
                            initial_t2 = previous_t2

                        t = initial_t2 - interval2 * action2
                        
                        # prevent t from exceeding the previous time step
                        thres = 999 if agent2_step_num == 0 else self.time_step_sequence[-1]
                        t = torch.ones(self.x.shape[0])*torch.tensor(int(max(0, min(t, thres))))

                        if (self.target_steps - agent2_step_num - 1) != 0:
                            interval2 = int(t / (self.target_steps - agent2_step_num - 1)) 
                    
                        self.agent2_x, kernel = self.DM.get_noisy_x(self.x, t, self.agent2_x0_t, self.y, self.agent2_et)
                        self.agent2_x0_t, self.agent2_et = self.DM.single_step_gibbsddrm(self.agent2_x, t)
                        
                        
                        observation2 = {
                            "image": self.agent2_x0_t[0].cpu(),  
                            "value": np.array([t]),
                            "remain": np.array([self.target_steps - agent2_step_num - 1])
                        }
                        self.time_step_sequence2.append(t.item())
                        agent2_step_num += 1
                
                # start from x0_t and t predicted by agent 1
                else:
                    observation2 = {
                        "image": self.agent2_x0_t[0].cpu(),  
                        "value": np.array([self.previous_t]), # or 999?
                        "remain": np.array([self.target_steps]),
                    }

                    # starting t is from agent 1 prediction
                    previous_t2 = self.previous_t.clone()
                    interval2 = self.interval
                    self.time_step_sequence2.append(t.item())

                    # agent 2 needs target step - 1 steps to finish the denoising 
                    # don't need to modify the first step from agent1
                    for agent2_step_num in range(self.target_steps - self.current_step_num -1):
                        print("remaining steps:", self.target_steps - agent2_step_num -1)
                        action2, _state = self.agent2.predict(observation2, deterministic=True)
                        
                        if agent2_step_num != 0 :
                            initial_t2 = previous_t2 - interval2 
                        else:
                            initial_t2 = previous_t2

                        t = initial_t2 - interval2 * action2
                        
                        # prevent t from exceeding the previous time step
                        thres = 999 if agent2_step_num == 0 else self.time_step_sequence[-1]
                        t = torch.tensor(int(max(0, min(t, thres))))

                        if (self.target_steps - agent2_step_num - 1) != 0:
                            interval2 = int(t / (self.target_steps - agent2_step_num - 1)) 
                    
                        self.agent2_x, kernel = self.DM.get_noisy_x(self.x, t, self.agent2_x0_t, self.y, self.agent2_et)
                        self.agent2_x0_t, self.agent2_et = self.DM.single_step_gibbsddrm(self.agent2_x, t)
                        
                        
                        observation2 = {
                            "image": self.agent2_x0_t[0].cpu(),  
                            "value": np.array([t]),
                            "remain": np.array([self.target_steps - agent2_step_num - 1])
                        }
                        self.time_step_sequence2.append(t.item())
                        agent2_step_num += 1
                    
                print("agent2 done, time_step_sequence2:", self.time_step_sequence2)
                self.time_step_sequence = self.time_step_sequence2
                    
            else:
                # print("uniform sampling")
                for i in range(self.target_steps - self.current_step_num - 1):
                    uniform_t = torch.ones(self.x.shape[0])*torch.tensor(int(t - self.interval - self.interval * i))
                    uniform_t = torch.ones(self.x.shape[0])*torch.tensor(max(0, min(uniform_t, 999)))
                    self.uniform_x, kernel = self.DM.get_noisy_x(self.x, uniform_t, self.uniform_x0_t, self.y, self.uniform_et)
                    self.uniform_x0_t, self.uniform_et = self.DM.single_step_gibbsddrm(self.uniform_x, uniform_t)
                    if self.adjust == False: # First subtask
                        self.time_step_sequence.append(uniform_t.item())

            
            
            # Run the remaining steps with uniform sampling to get pivot_x_0|t for reward calculation
            # for i in range(self.target_steps - self.current_step_num - 1):
            #     uniform_t = torch.tensor(int(initial_t - self.old_interval - self.old_interval * i))
            #     uniform_t = torch.tensor(max(0, min(uniform_t, 999)))
            #     self.uniform_pivot_x = self.DM.get_noisy_x(uniform_t, self.pivot_x0_t, self.pivot_et)
            #     self.pivot_x0_t, _,  self.pivot_et = self.DM.single_step_ddnm(self.uniform_pivot_x, self.y, uniform_t, self.classes)

        self.cnt += 1
        # print("cnt = ", self.cnt)
        # Finish the episode if denoising is done
        done = (self.current_step_num == self.target_steps - 1) or not self.adjust
        # Calculate reward
        reward, ssim, psnr, ddim_ssim, ddim_psnr, pivot_ssim, pivot_psnr = self.calculate_reward(done)
        # Save figure
        # if done:
        #     self.DM.postprocess(self.x0_t, self.x_orig, self.data_idx)
        info = {
            'ddim_t': self.uniform_steps[self.current_step_num],
            't': t,
            'reward': reward,
            'ssim': ssim,
            'psnr': psnr,
            'pivot_ssim': pivot_ssim,
            'pivot_psnr': pivot_psnr,
            'ddim_ssim': ddim_ssim,
            'ddim_psnr': ddim_psnr,
            'ddnm_ssim': self.ddnm_ssim,
            'ddnm_psnr': self.ddnm_psnr,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence,
            'threshold': self.final_threshold,
        }
        # print('info:', info)
        if self.adjust:
            observation = {
                "image": self.x0_t[0].cpu(),
                # "image2": self.Apy[0].cpu(),
                "value": np.array([t]),
                "remain": np.array([self.target_steps - self.current_step_num - 1])
            }
        else:
            observation = {
                "image":  self.x0_t[0].cpu(),  
                # "value": np.array([t])
            }
        self.current_step_num += 1
        torch.cuda.empty_cache()  # Clear GPU cache
        return observation, reward, done, truncate, info

    def get_ssim_psnr(self, x, orig):
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        return ssim, psnr

    def calculate_reward(self, done):
        reward = 0
        orig = inverse_data_transform(self.DM.config, self.x_orig[0]).to(self.DM.device)
        if done and self.adjust: # Second subtask done
            x = inverse_data_transform(self.DM.config, self.x0_t[0]).to(self.DM.device)
        elif self.agent2:
            x = inverse_data_transform(self.DM.config, self.agent2_x0_t[0]).to(self.DM.device)
        else: # First subtask or Second subtask not done
            x = inverse_data_transform(self.DM.config, self.uniform_x0_t[0]).to(self.DM.device)
        # pivot_x = inverse_data_transform(self.DM.config, self.pivot_x0_t[0]).to(self.DM.device)

        ssim, psnr = self.get_ssim_psnr(x, orig)
        # pivot_ssim, pivot_psnr = self.get_ssim_psnr(pivot_x, orig)


        # reward += 0.5 * (psnr - pivot_psnr) / (27.46 - 26.82)
        # reward += 0.5 * (ssim - pivot_ssim) / (0.01)
        if self.adjust == False: # First subtask
            self.pivot_ssim = self.ddim_ssim
            self.pivot_psnr = self.ddim_psnr
            
        if ssim > self.pivot_ssim:
            reward += ssim / self.pivot_ssim
        else:
            reward -= self.pivot_ssim / ssim
        if psnr > self.pivot_psnr:
            reward += psnr / self.pivot_psnr
        else:
            reward -= self.pivot_psnr / psnr

        if not done:
            reward /= self.target_steps

        # Intermediate reward
        '''if not done:# and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            if self.ddnm_psnr > pivot_psnr:
                reward += 0.5/self.target_steps * (psnr - pivot_psnr) / (self.ddnm_psnr - pivot_psnr)
            else:
                reward += 0.5/self.target_steps * psnr / pivot_psnr
            if self.ddnm_ssim > pivot_ssim:
                reward += 0.5/self.target_steps * (ssim - pivot_ssim) / (self.ddnm_ssim - pivot_ssim)
            else:    
                reward += 0.5/self.target_steps * ssim / pivot_ssim
            # reward += 0.5/self.target_steps * (psnr / self.ddim_psnr)  
            # reward += 0.5/self.target_steps * (ssim / self.ddim_ssim)  
            
        # Sparse reward
        if done:# and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            if self.ddnm_psnr > pivot_psnr:
                reward += 0.5 * (psnr - pivot_psnr) / (self.ddnm_psnr - pivot_psnr)
            else:
                reward += 0.5 * psnr / pivot_psnr
            if self.ddnm_ssim > pivot_ssim:
                reward += 0.5 * (ssim - pivot_ssim) / (self.ddnm_ssim - pivot_ssim)
            else:    
                reward += 0.5 * ssim / pivot_ssim
            # reward += 0.5 * (psnr / self.ddim_psnr)
            # reward += 0.5 * (ssim / self.ddim_ssim)'''

        # print('ssim:', ssim, 'psnr:', psnr, 'pivot_ssim:', pivot_ssim, 'pivot_psnr:', pivot_psnr, 'ddnm_ssim:', self.ddnm_ssim, 'ddnm_psnr:', self.ddnm_psnr, 'reward:', reward)
        return reward, ssim, psnr, self.ddim_ssim, self.ddim_psnr, self.pivot_ssim, self.pivot_psnr
    
    def render(self, mode='human', close=False):
        # This could visualize the current state if necessary
        pass

    def set_adjust(self, adjust):
        self.adjust = adjust
        print(f"Set adjust to {adjust}")
