import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):       ## Guide it to the original data x_0 not the start point
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -      ### Reverse process SUBTRACTING the noise
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise    ### the noise is from model(x, cond, t)
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):     ## The new mean and variance to sample the previous image with , x_t is the current noisy image ... not the end point!?
        print("x_start",x_start)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +     # extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +## posterior_mean_coef2 = 1 - posterior_mean_coef1 - So a linear combination
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t            ## GUIDED MEAN of the co-ords ... This is an interpoloation no!?!?
        ) 
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)   ## UNGUIDED VARIANCE ... 
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        print("self.model(x, cond, t)",self.model(x, cond, t))
 
        print("cond",cond) 
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))   ## NOISE Adjusted by the conditions!?!?!!?
        ## x_start=x_recon  --- x_recond is apparently as estimate of the training data!?!?!!?!? - Its adjusts x according to the noise from the conditions 
        if self.clip_denoised: 
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        ## GUIDE the mean and var based on x_start ... x_recon 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(      
                x_start=x_recon, x_t=x, t=t)                                            ## x_start=x_recon
        return model_mean, posterior_variance, posterior_log_variance                   ## Return the GUIDED mean to sample from + GUIDED variances - shifted maybe due to guidance 

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)   ## Grab the conditionally GUIDED mean and variance f
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise    ### An adjusted Mean!!  Advanced Repar Trick
 
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):         ### Where do the conditions come from? Can be overwritten in apply_conditioning!? 
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)   ## Creates 10 x 128 random **NOISY*** points - the 10 samples you see in the images :) 
        x = apply_conditioning(x, cond, self.action_dim)    ## Make the random numbers have a certain start and end point

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):                                          ### 64 starts at and then goes backwards
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)      ## Sample but FIX the start and end points 
  
            progress.update({'t': i})
 
            if return_diffusion: diffusion.append(x)

        progress.close()
 
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, **kwargs):    ###  conditional_sample  !!
        '''
            conditions : [ (time, state), ... ]
        '''
        # 3pointer if you want to HARD CODE the SAMPLINg conditions here
   
        device = self.betas.device
        batch_size = len(cond[0])             ### First sight of the cond
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):    ## add the noise -- ####   forward process!! ####
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
 
    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        #print("x_start",x_start)
        #print("noise",noise)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)   ## get the forward noise
        #print("x_noisy",x_noisy)
        # print("cond",cond)

        #shape = cond.shape
  
        ##  AttemptedFixedConditions = torch.full(shape, -0.3, device='cuda:0')
        # Set the second column to 0.4
        ## AttemptedFixedConditions[:, 1] = 0.4
        ## cond = AttemptedFixedConditions

        ## print("cond",cond)
  
        #print("self.action_dim",self.action_dim)
        ## Conditions are like where the wall is!? Giving equiv of negative rewards ?
        ## No I think Cond = start and End points I think ...
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)      # noisy but with start and end points fixedd 

        x_recon = self.model(x_noisy, cond, t)     
        #print("x_recon",x_recon)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)     
        #print("x_recon",x_recon)

        assert noise.shape == x_recon.shape
 
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)    ## Loss function - how well does the noisy image reflect the one - to train the NN on - so it can take off noise
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)    ### Calls p loss

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

