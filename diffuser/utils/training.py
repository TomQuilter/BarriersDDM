import os
import copy
import numpy as np
import torch
import einops
import pdb
 
#tensor = torch.tensor([1, 2, 3])
#tensor = tensor.to(device='cuda:0')

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
 
def generate_conditions(num_rows):
    # Adjust for two middle points
    first_mid_index = num_rows // 3  # First third for the first middle point
    second_mid_index = 2 * num_rows // 3  # Two thirds for the second middle point
    end_index = num_rows - 1

    # Initialize conditions with specific points, including a second middle point
    conditions = {
        0: torch.tensor([[-0.7, -0.7, 1.0000, -0.8000]], device='cuda:0'),  # Blue dot
        first_mid_index: torch.tensor([[-0.7, 0.9, -1.0000, -0.8000]], device='cuda:0'),  # First Green Dot
        second_mid_index: torch.tensor([[0.7, 0.9, 1.0000, -0.8000]], device='cuda:0'),  # Second Green Dot (new)
        end_index: torch.tensor([[0.7, -0.7, -1.0000, -0.8000]], device='cuda:0')  # Red Dot
    }  

    # Interpolate between start and first middle
    x_step_1 = (conditions[first_mid_index][0][0] - conditions[0][0][0]) / first_mid_index
    y_step_1 = (conditions[first_mid_index][0][1] - conditions[0][0][1]) / first_mid_index
    for i in range(1, first_mid_index):
        x = conditions[0][0][0] + i * x_step_1
        y = conditions[0][0][1] + i * y_step_1
        conditions[i] = torch.tensor([[x, y, 1.0000, -0.8000]], device='cuda:0')

    # Interpolate between first and second middle
    x_step_2 = (conditions[second_mid_index][0][0] - conditions[first_mid_index][0][0]) / (second_mid_index - first_mid_index)
    y_step_2 = (conditions[second_mid_index][0][1] - conditions[first_mid_index][0][1]) / (second_mid_index - first_mid_index)
    for i in range(first_mid_index + 1, second_mid_index):
        x = conditions[first_mid_index][0][0] + (i - first_mid_index) * x_step_2
        y = conditions[first_mid_index][0][1] + (i - first_mid_index) * y_step_2
        conditions[i] = torch.tensor([[x, y, -1.0000, -0.8000]], device='cuda:0')

    # Interpolate between second middle and end
    x_step_3 = (conditions[end_index][0][0] - conditions[second_mid_index][0][0]) / (end_index - second_mid_index)
    y_step_3 = (conditions[end_index][0][1] - conditions[second_mid_index][0][1]) / (end_index - second_mid_index)
    for i in range(second_mid_index + 1, end_index):
        x = conditions[second_mid_index][0][0] + (i - second_mid_index) * x_step_3
        y = conditions[second_mid_index][0][1] + (i - second_mid_index) * y_step_3
        conditions[i] = torch.tensor([[x, y, 1.0000, -0.8000]], device='cuda:0') 

    return conditions

# Example usage:
num_rows = 10  # Set the number of rows as needed
conditions = generate_conditions(num_rows)
 
def generate_conditionstwopoints(num_rows):
# Fixed indexes based on the original specification, adjusted dynamically
    mid_index = num_rows // 2 
    end_index = num_rows - 1

    # Initialize conditions with specific points, dynamically setting the second point
    conditions = {
        0: torch.tensor([[-0.7, -0.7, 1.0000, -0.8000]], device='cuda:0'),        ## Blue dot
        mid_index: torch.tensor([[-0.7, 0.9, -1.0000, -0.8000]], device='cuda:0'),## Green Dot
        end_index: torch.tensor([[0.7, 0.9, 1.0000, -0.8000]], device='cuda:0')   ## Red Dot
    } 

    # Calculate step sizes for interpolation between the first and mid point, and mid and end point
    x_step_1 = (conditions[mid_index][0][0] - conditions[0][0][0]) / mid_index
    y_step_1 = (conditions[mid_index][0][1] - conditions[0][0][1]) / mid_index
    x_step_2 = (conditions[end_index][0][0] - conditions[mid_index][0][0]) / (end_index - mid_index)
    y_step_2 = (conditions[end_index][0][1] - conditions[mid_index][0][1]) / (end_index - mid_index)

    # Interpolate for indices 1 to mid_index - 1
    for i in range(1, mid_index):
        x = conditions[0][0][0] + i * x_step_1
        y = conditions[0][0][1] + i * y_step_1
        conditions[i] = torch.tensor([[x, y, 1.0000, -0.8000]], device='cuda:0')

    # Interpolate for indices mid_index + 1 to end_index - 1
    for i in range(mid_index + 1, end_index):
        x = conditions[mid_index][0][0] + (i - mid_index) * x_step_2
        y = conditions[mid_index][0][1] + (i - mid_index) * y_step_2
        conditions[i] = torch.tensor([[x, y, -1.0000, -0.8000]], device='cuda:0') 

    def print_conditions(conditions):
        for key in sorted(conditions.keys()):
            print(f"Index {key}: {conditions[key]}")

    # Call the function to print the conditions
    print_conditions(conditions) 

    return conditions

def cycle(dl):
    while True:
        for data in dl:
            yield data
 
class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        )) 
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(                      # 3pointer 
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True  #### Grab ONE trajectory from the 790,000 Batches RANDOMLY ... If Shuffle = True
        )) 
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):                          ### TRAIN!!
 
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):     ## gradient_accumulate_every??
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)         ###  Calls LOSS and PLoss for this batch
                loss = loss / self.gradient_accumulate_every   # Normalise?
                loss.backward()                               ###  Gradiant Descent via Pytorch here! 

            self.optimizer.step()                             ### Update the weights - x = x + learning rate * grad ....
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')      ## {Print Trianing Results}

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)     ## Print Reference 

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)           ## Render Samples 

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')   ## SAVE THE RESULTS HERE
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#
        


    def render_reference(self, batch_size=10):    ## Plot the REFERENCE images!
        '''
            renders training points
        '''
 
        ## get a temporary dataloader to load a ***single batch*** of 50 from all possible batches
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))    
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
  
        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')   ## Initial Reference images of paths
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint                              #3pointer 
            batch = self.dataloader_vis.__next__()                  
            conditions = to_device(batch.conditions, 'cuda:0')     ## Getting a random trajectry (Batch) from 790000, then grabbing the start and end point from that trajectory                                             ## grabbing New conditions, or Old conditions or ???
 
            print("conditions = ",conditions)
 
            #conditions = {
            #0: np.array([-0.9, -0.9, -1.0, -1.0], dtype=np.float32),
            #64: np.array([-0.5, -0.5, 1.0, 0.6], dtype=np.float32), 
            #127: np.array([-0.1, -0.1, 1.0, 0.6], dtype=np.float32)
            #}    
 
            ### HardCode The SAMPLING conditions for sample rendering 
   
            #conditions = { 
            #0: torch.tensor([[-0.7, -0.7, -0.8000, -0.8000]], device='cuda:0'),      ## Blue dot
            #32: torch.tensor([[-0.7, 0.1, -1.0000, -0.8000]], device='cuda:0'),
            #64: torch.tensor([[-0.7, 0.9, -1.0000, -0.8000]], device='cuda:0'),      ## green Dot
            #96: torch.tensor([[0.00, 0.9, -1.0000, -0.8000]], device='cuda:0'),
            #127: torch.tensor([[0.7, 0.9, -1.0000, -0.8000]], device='cuda:0')      ## Red Dot
            #}   
            
            conditions = generate_conditions(128)      ## Yes but ONLY overwrite the Trajectory Data ... in the helpers FN ... 
       
            ## repeat each item in **conditions** `n_samples` times for each of the n samples - 10 here          ### ConditionsGBI###
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )
  
            print("conditions = ",conditions)
 
            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)      ### conditional_sample GBI ### 
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]   

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)                         ## Save images based on observations
