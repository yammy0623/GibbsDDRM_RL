import glob
import logging
import os
import random
import shutil
import time
from datetime import datetime

import lpips
import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
from torch.nn.parallel import DataParallel
import tqdm
from datasets import data_transform, get_dataset, inverse_data_transform
from functions.ckpt_util import download, get_ckpt_path
from functions.denoising import sample_gibbsddrm
from guided_diffusion.script_util import (args_to_dict, classifier_defaults,
                                          create_classifier, create_model)
from guided_diffusion.unet import UNetModel
from models.diffusion import Model
from pytz import timezone
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torch.utils import tensorboard
from torch.utils.tensorboard.writer import SummaryWriter

def scale_up(x_orig):
    x_orig_copy = torch.cat([x_orig, x_orig], dim=0)
    return x_orig_copy

def scale_down(x_orig):
    device = x_orig.device
    x_orig = torch.index_select(x_orig, dim=0, index=torch.tensor([0], device=device))
    return x_orig

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class MyDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        dataset, test_dataset = get_dataset(args, config)
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)
        self.dataset = dataset
        self.test_dataset = test_dataset 

        print(f'Dataset has size {len(test_dataset)}')   
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            elif self.config.data.dataset == "FFHQ":
                name = 'ffhq'
            else:
                raise ValueError
            if name != 'celeba_hq' and name != 'ffhq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            elif name == "ffhq":
                ckpt = os.path.join(self.args.exp, "logs/ffhq/ffhq_10m.pt")
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()

            if self.config.data.dataset == "FFHQ":
                ckpt = os.path.join(self.args.exp, "logs/ffhq/ffhq_10m.pt")
            elif self.config.data.dataset == "AFHQ":
                ckpt = os.path.join(self.args.exp, "logs/afhq/afhqdog_p2.pt")
            else:
                if self.config.model.class_cond:
                    ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                    if not os.path.exists(ckpt):
                        download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
                else:
                    ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                    if not os.path.exists(ckpt):
                        download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn
        else:
            model=None
        self.model = model
        self.cls_fn = cls_fn
        
 
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
        self.val_loader = val_loader
        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg == "deblur_arbitral":
            
            # Since H_funcs is 
            from functions.svd_replacement import DeblurringArbitral2D

            conv_type = config.deblur.conv_type

        else:
            print("ERROR: degradation type not supported")
            quit()
        sigma_0 = 2 * config.deblur.sigma_0 # to account for scaling to [-1, 1]
        self.sigma_0 = sigma_0
        
        print(f'Start from {args.subset_start}')
        # idx_init = args.subset_start
        # idx_so_far = args.subset_start
        # avg_psnr = 0.0
        # pbar = tqdm.tqdm(val_loader)

        # Make directory which stores result images
        dt_now = datetime.now(timezone('Asia/Tokyo'))
        dt_str = dt_now.strftime('%Y_%m%d_%H%M%S')
        image_folder = self.args.image_folder + dt_str
        os.makedirs(image_folder, exist_ok=False)
        # save config file
        config_path = os.path.join("configs", self.args.config)
        shutil.copyfile(config_path, os.path.join(image_folder, self.args.config))

        if config.logger.enable_log:
            # Tensorboard
            config.logger.writer = SummaryWriter(image_folder)
            config.logger.image_folder = image_folder
            config.logger.inverse_data_transform = inverse_data_transform

        # batch_size = config.sampling.batch_size
        batch_size = 2

        kernel_batch = DeblurringArbitral2D.get_blur_kernel_batch(batch_size, config.deblur.kernel_type, self.device)
        kernel_uncert_batch = \
            DeblurringArbitral2D.corrupt_kernel_batch(kernel_batch, \
                                                                config.deblur.kernel_corruption, \
                                                                config.deblur.kernel_corruption_coef)

        H_funcs = DeblurringArbitral2D(kernel_batch, config.data.channels, self.config.data.image_size, self.device, conv_type)
        H_funcs_uncert = DeblurringArbitral2D(kernel_uncert_batch, config.data.channels, self.config.data.image_size, self.device, conv_type)
        self.H_funcs = H_funcs
        self.H_funcs_uncert = H_funcs_uncert
        # Process data

    def preprocess(self, x_orig, idx_so_far):
        args, config = self.args, self.config
        H_funcs = self.H_funcs
        sigma_0 = self.sigma_0
        b = self.betas
    
        # batch_size = x_orig.shape[0]

        x_orig = x_orig.to(self.device)
        x_orig = data_transform(self.config, x_orig)

        # print("x_orig", x_orig.shape)
        x_orig = scale_up(x_orig)
        
        # print("x_orig", x_orig.shape)
        y_0 = H_funcs.H(x_orig)            
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
        # print("x0", x_orig.shape)
        # print("y0", y_0.shape)
        
        # Save images to the directory
        # for i in range(len(y_0)):
        #     tvu.save_image(
        #         inverse_data_transform(config, y_0[i].view(self.config.data.channels, H_funcs.out_img_dim, H_funcs.out_img_dim)), os.path.join(image_folder, f"y0_{idx_so_far + i}.png")
        #     )
        #     tvu.save_image(
        #         inverse_data_transform(config, x_orig[i]), os.path.join(image_folder, f"orig_{idx_so_far + i}.png")
        #     )

        ##Begin GibbsDDRM
        x = torch.randn(
            y_0.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        # print("x", x.shape)
        self.y_0s = []
        self.kernels = []
        # not sure whether it could be A_inv_y
        start_T = 990
        U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, b,  y_0, sigma_0, self.y_0s, x, start_T=start_T, init=True)
        H_inv_y = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())

        # print("device", x.device)
        x = scale_down(x)
        y_0 = scale_down(y_0)
        x_orig = scale_down(x_orig)
        H_inv_y = scale_down(H_inv_y)
        return x, y_0, x_orig, H_inv_y

    def postprocess(self, x, idx_so_far):
        config = self.config
        x = [inverse_data_transform(config, y) for y in x]
        tvu.save_image(
            x[0], os.path.join(self.args.image_folder, f"{idx_so_far}_{0}.png")
        )

    
    def single_step_gibbsddrm(self, x, t):
        x = scale_up(x)
        t = scale_up(t)
        # print("single step x", x.shape)
        # print("single step t", t.shape)

        xt = x.to('cuda')
        t = torch.tensor(t).to(xt.device)
        at = compute_alpha(self.betas, t.long())
        model = self.model
        x0_t, et = est_x0_t(model, xt, t, at)

        x0_t = scale_down(x0_t)
        et = scale_down(et)
        return x0_t.cpu(), et.cpu()
    
    def get_noisy_x(self, x, next_t, x0_t, y_0, et, first_step=False, initial=False):
        x = scale_up(x)
        next_t = scale_up(next_t)
        x0_t = scale_up(x0_t)
        y_0 = scale_up(y_0)
        et = scale_up(et)

        next_t = next_t.to('cuda')
        x0_t = x0_t.to('cuda')
        et = et.to('cuda')
        sigma_0 = self.sigma_0
        model = self.model
        
        H_funcs = self.H_funcs_uncert
        etaB=self.config.deblur.etaB
        etaA=self.config.deblur.etaA
        etaC=self.config.deblur.etaC
        etaD = self.config.deblur.etaD
        max_steps = 999
        b = self.betas

        cls_fn=self.cls_fn, 
        config=self.config

        at_next = compute_alpha(b, next_t.long())

        
        if next_t[0].to('cpu') < int(max_steps * 0.7):
            U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, b,  y_0, sigma_0, self.y_0s, x=x, x_0=x0_t)
        else:
            U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, b,  y_0, sigma_0, self.y_0s, x=x)


        xt_next = update_x(H_funcs, U_t_y, sigma_0, singulars, Sigma, Sig_inv_U_t_y, x0_t, et, at_next, etaA, etaB, etaC, etaD)

        # Hupdate
        enable_Hupdate = (next_t[0].to('cpu') <= max_steps * config.deblur.Hupdate_start)
        if enable_Hupdate:

            for i_Hupdate in range(config.deblur.iter_Hupdate):
                # first step
                if first_step:
                    continue
                
                x0_t_next, et_next = est_x0_t(model, xt_next, next_t, at_next)

                if config.deblur.alg_Hupdate == "optim":
                    H_funcs.update_H_optim(y_0, x0_t_next, n_iter=config.deblur.iter_optim, lr=float(config.deblur.lr_Hupdate), \
                        reg_H_gamma=config.deblur.reg_H_gamma, reg_H_type = config.deblur.reg_H_type)
                elif config.deblur.alg_Hupdate == "langevin": # linear operator's parameter update of GibbsDDRM
                    H_funcs.update_H_langevin(y_0, x0_t_next, n_iter=config.deblur.iter_optim, lr=float(config.deblur.lr_Hupdate), \
                        reg_H_gamma=config.deblur.reg_H_gamma, reg_H_type = config.deblur.reg_H_type)

                if i_Hupdate == (config.deblur.iter_Hupdate - 1) and config.deblur.resample_after_Hupdate is False:
                    continue

                U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs,b, y_0, sigma_0, self.y_0s, x, x_0=x0_t)                    
                xt_next = update_x(H_funcs, U_t_y, sigma_0, singulars, Sigma, Sig_inv_U_t_y, x0_t, et, at_next, etaA, etaB, etaC, etaD)

        xt_next = scale_down(xt_next)
        return xt_next.to('cpu'), H_funcs.kernel.detach().clone().to('cpu')


    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.config.deblur.timesteps
        seq = range(0, self.num_timesteps, skip)

        x_init = x

        for i_ddrm in range(self.config.deblur.iter_DDRM):

            x = sample_gibbsddrm(x_init, seq, model, self.betas, H_funcs, y_0, sigma_0, \
                    etaB=self.config.deblur.etaB, etaA=self.config.deblur.etaA, etaC=self.config.deblur.etaC, etaD = self.config.deblur.etaD, cls_fn=cls_fn, classes=classes, 
                    config=self.config)

            if self.config.logger.enable_log:
                # Log DDRM output
                x_on_cpu = self.config.logger.inverse_data_transform(self.config, x[0][-1])
                x_on_cpu = x_on_cpu.to("cpu").detach()
                self.config.logger.writer.add_images("DDRM output", x_on_cpu, i_ddrm)
                # Log kernel
                kernel_on_cpu = H_funcs.kernel[:, None, :, :].repeat(1, 3, 1, 1).to("cpu").detach()
                self.config.logger.writer.add_images("Refined kernel", torch.abs(kernel_on_cpu)/torch.max(torch.abs(kernel_on_cpu)), i_ddrm)
                
        if last:
            x = [x[0][-1]]
        return x



# sample_gibbsddrm

def calc_vars_for_xupdate(H_funcs, b, y_0, sigma_0, y_0s, x=None, x_0 = None, start_T=None, init=False):
    """
        calculate variables that are used when samling x_t.
        As these variables depends on linear operator's parameters (phi), 
        they must be updated after the update of phi.

    """
        # x 是定值，USV會變
   
    bsz = y_0.shape[0]
    # print("device", y_0.device)
    if H_funcs.conv_shape == "same_interp":
        if x_0 is not None:
            # print("device", x_0.device)
            y_0_interp = H_funcs.interp_y_0(y_0, x_0, sigma_0)
        else:
            
            y_0_interp = H_funcs.interp_y_0(y_0, y_0, sigma_0)
        U_t_y = H_funcs.Ut(y_0_interp)
        y_0s.append(y_0_interp)
    else:
        U_t_y = H_funcs.Ut(y_0)
    _dim_all_singulars = U_t_y.view(U_t_y.shape[0], -1).shape[1]

    #setup vectors used in the algorithm
    singulars = H_funcs.singulars()
    Sigma = torch.zeros(bsz, _dim_all_singulars, device=x.device)
    Sigma[:, :singulars.shape[-1]] = singulars
    Sig_inv_U_t_y = U_t_y / singulars[:, :U_t_y.shape[-1]]

    init_y = None
    if init:
        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * start_T).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where((singulars * largest_sigmas[:, 0, 0, 0][:, None]) > sigma_0)
        inv_singulars_and_zero = torch.zeros(singulars.shape).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(bsz, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        #   if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], singulars.shape[-1], dtype=U_t_y.dtype).to(x.device)
        init_y[large_singulars_index] = U_t_y[large_singulars_index] / singulars[large_singulars_index]
        remaining_s = (largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2)
        remaining_s = remaining_s.clamp_min(0.0).sqrt()
        V_t_x_init = H_funcs.Vt(x)
        init_y = init_y + remaining_s * V_t_x_init
        init_y = init_y / largest_sigmas.view(largest_sigmas.shape[0], -1)


    return U_t_y, singulars, Sigma, Sig_inv_U_t_y,init_y

def update_x(H_funcs, U_t_y, sigma_0, singulars, Sigma, Sig_inv_U_t_y, x0_t, et, at_next, etaA, etaB, etaC, etaD):

    """
        perform the modified DDRM steps defined in Eq. (9) in the paper, 
        x_t is sampled from p_theta (x_t | x_{t+1}, phi, y)

        Returns:
            xt_next : the sampled x_t
        
    """
    

    bsz = x0_t.shape[0]
    #variational inference conditioned on y
    # sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
    sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
    # xt_mod = xt / at.sqrt()[0, 0, 0, 0]
    # V_t_x = H_funcs.Vt(xt_mod)
    # SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
    V_t_x0 = H_funcs.Vt(x0_t)
    Sigma = torch.tensor(Sigma, device=U_t_y.device) 
    # print("Sigma", Sigma.device)
    # print("U_t_y", U_t_y.device)
    # print("V_t_x0", V_t_x0.device)   
    SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

    falses = torch.zeros(bsz, V_t_x0.shape[1] - singulars.shape[-1], dtype=torch.bool, device=x0_t.device)

    cond_before_lite = (singulars * sigma_next > sigma_0) * (singulars > 1e-10)
    cond_after_lite =  (singulars * sigma_next < sigma_0) * (singulars > 1e-10)

    cond_before = torch.hstack((cond_before_lite, falses))
    cond_after  = torch.hstack((cond_after_lite, falses))

    std_nextD = sigma_next * etaD
    sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextD ** 2)

    # std_nextA = sigma_next * etaA
    # sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)

    diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

    #missing pixels        
    Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextD * H_funcs.Vt(torch.randn_like(x0_t))

    #less noisy than y (after)
    coef_A = sigma_next * etaA
    coef_C = sigma_next * etaC

    update_A = H_funcs.Vt(et) * cond_after_lite
    update_C = ((U_t_y - SVt_x0) / sigma_0) * cond_after_lite

    corr_coef = torch.abs((update_A * torch.conj(update_C)).sum(-1)) / (update_A.norm(dim=-1) * update_C.norm(dim=-1) + 1e-10)
    std_coef = sigma_next * torch.sqrt(1 - etaA**2-etaC**2 - 2 * etaA*etaC*corr_coef)
    Vt_xt_mod_next[cond_after] = \
                V_t_x0[cond_after] + coef_A * update_A[cond_after] + coef_C * update_C[cond_after] + (std_coef[:, None] * H_funcs.Vt(torch.randn_like(x0_t)))[cond_after]

    #noisier than y (before)
    Vt_xt_mod_next[cond_before] = \
                (Sig_inv_U_t_y[cond_before_lite] * etaB + (1 - etaB) * V_t_x0[cond_before] + diff_sigma_t_nextB * H_funcs.Vt(torch.randn_like(x0_t))[cond_before_lite])

    #aggregate all 3 cases and give next predictionz
    xt_mod_next = H_funcs.V(Vt_xt_mod_next)
    xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x0_t.shape)

    return xt_next


def est_x0_t(model, xt, t, at):
    et = model(xt, t)
    if et.size(1) == 6:
        et = et[:, :3]
    x0_t = (xt - et *(1-at).sqrt()) / at.sqrt()
    return x0_t, et