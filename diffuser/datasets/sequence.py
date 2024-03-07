from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

def generate_string(N):
    # Base part of the string, including the initial return statement and opening brace
    string_parts = [""]
    
    # Loop through each number from 0 to N-2 to generate the middle part of the string
    for i in range(N-1):
        string_parts.append(f"    {i}: observations[{i}],\n")
    
    # Add the last special line for self.horizon - 1
    string_parts.append(f"    self.horizon - 1: observations[-1],\n")
    
    # Closing brace for the dictionary
    string_parts.append("")
     
    # Join all parts into a single string
    return "".join(string_parts) 

# Example usage
#N = 3  # You can change this value to generate a string for any N rows
#result_string = generate_string(5)
#print(result_string)
 
class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes  ##  n_episodes = 1566 
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):  ## path lengths - array([ 267, 1199,  388, ..., 1375,  276,  583])
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        ''' 
            condition on current observation for planning
        '''
        1 == 1 

        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]  ## get some index of the path - start and end point

        observations = self.fields.normed_observations[path_ind, start:end]    ### Get the original NOMRALISED Obs or/AND! the Diffused ones?
        actions = self.fields.normed_actions[path_ind, start:end] 
        #print("conditions = ", "")
        conditions = self.get_conditions(observations)        ##3pointer
  
        #conditions = {
        #0: np.array([-0.9, -0.9, -1.0, -1.0], dtype=np.float32),
        #64: np.array([-0.5, -0.5, 1.0, 0.6], dtype=np.float32), 
        #127: np.array([-0.1, -0.1, 1.0, 0.6], dtype=np.float32)
        #}  
        #print("conditions = ", conditions)
  
        trajectories = np.concatenate([actions, observations], axis=-1)        ### Combine the ACTION and OBS
        batch = Batch(trajectories, conditions)                                ### Batch the dataset!!! inc the conditions for the TRAINING Data!  
        return batch  
  
class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):   ### Get Conditions For ***Training***  
        # TQ - Maybe Grab a training sample Traj - Grab its first and last points!??! think thats what Avirup said, then do it for another oanother one. Loss based off the different?
        ## Use these for sampling ... 

        ## Should now be the WHOLE trajectory really - thats easily done no? 

        '''
            condition on both the current observation and the last observation in the plan   ## OK!!!   # 3pointer 
        ''' 
 
        conditions = {k: observations[min(k, len(observations) - 1)] for k in range(self.horizon)}
 
        # Use a dictionary comprehension to construct the dictionary dynamically.
        # This approach avoids the direct use of `eval` for better safety and readability.
        #conditions = {k: observations[min(k, len(observations) - 1)] for k in keys}

        # print("Conditions", conditions)
  
        return conditions
  
class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
