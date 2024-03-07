import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()     ## Get the DATA From the gym environment

    #print("Type of dataset:", type(dataset))

    import random

  
    #dataset['observations'][:, 0] = random.random()  
    #dataset['observations'][:, 1] = random.random() 
    # dataset['observations'][:, 2] = 0       
    #dataset['observations'][:, 3] = 0    
    #print("dataset",dataset)  
 
    # Call the function to write the dataset to the CSV file
    csv_file = 'outputWithGoalsLDS.csv'
    write_dataset_to_csv(dataset, csv_file, perform=False)
        
    print("dataset['observations'][0, 0]",dataset['observations'][0, 0])
    print("dataset['observations'][0, 0]",dataset['observations'][0, 1])
    print("dataset['observations'][0, 0]",dataset['observations'][1, 0])   
    print("dataset['observations'][0, 0]",dataset['observations'][1, 1])

    dataset['observations'][0, 0] = 0.4
    dataset['observations'][0, 1] = 0.8   
  
    observation_size = dataset['observations'].shape[0]

    print("dataset == ",dataset)  

    print("dataset['observations'].shape[0]",dataset['observations'].shape[0])

            # Create a new array with an additional column
    #new_shape = (dataset.observations.shape[0], dataset.observations.shape[1] + 1)
    #new_observations = np.ones(new_shape) * np.nan  # Initialize with NaN or another placeholder value
   
    # Copy existing data to the new array
    #new_observations[:, :-1] = observations

    # Now, iterate and set the value of the new column
    #for row in range(new_observations.shape[0]):
    #    new_observations[row, -1] = 1.2  # Assign 1.2 to the last column for each row

    # Update the original variable if desired
    #observations = new_observations

    def custom_random_uniform(low1, high1, low2, high2):
        # Decide which range to use
        if random.random() < 0.5:  # 50% chance for each range
            return random.uniform(low1, high1)
        else:
            return random.uniform(low2, high2)
     
    # print(custom_random_uniform(0, 0.3, 0.7, 1))
       
    for row in range(int(observation_size)):    ## int(observation_size * 0.9) // 10
        1==1    
        #dataset['actions'][row, 0] = random.uniform(1, 1.0001) # random.random()
        #dataset['actions'][row, 1] = dataset['actions'][row, 0] # random.uniform(1, 1.0001) # random.random() 
        RandomNumber = random.uniform(0.4, 0.6) 
        dataset['observations'][row, 2] = RandomNumber - 0.5  ## 0.33*random.uniform(0, 1) - 0.5 # random.random()    
        
        if 0 <= RandomNumber <= 0.2:
            y_range_start, y_range_end = 0, 1
            y = random.uniform(y_range_start, y_range_end)
        elif 0.2 < RandomNumber <= 0.4:
            y = custom_random_uniform(0, 0.2, 0.8, 1)
        elif 0.4 < RandomNumber <= 0.6:
            y_range_start, y_range_end = 0, 0.6
            y = custom_random_uniform(0, 0.4, 0.9, 1)     ##             y = custom_random_uniform(0, 0.6, 0.8, 1)
        elif 0.6 < RandomNumber <= 0.8:  
            y = custom_random_uniform(0, 0.2, 0.8, 1)
        elif 0.8 < RandomNumber <= 1:  # This condition is supposed to be always true if x is between 0 and 0.8
            y_range_start, y_range_end = 0, 1
            y = random.uniform(y_range_start, y_range_end)
        else: 
            # If x is not in the range [0, 0.8], we don't draw y
            y_range_start, y_range_end = None, None
            y= 0
      
        dataset['observations'][row, 3] = y - 0.5            
        #dataset['observations'][row, 2] = dataset['observations'][row, 0]
        #dataset['observations'][row, 3] = dataset['observations'][row, 1] 
        #dataset['observations'][row, 0] = (row/int(observation_size)) * 0.1 # random.uniform(1, 2) # random.uniform(1, 2) # random.random() 
        #dataset['observations'][row, 1] = (row/int(observation_size)) * 0.1  
        #dataset['observations'][row, 0] = 1+ dataset['observations'][row, 0] + (row/int(observation_size)) * 0.1 # random.uniform(1, 2) # random.uniform(1, 2) # random.random() 
        #dataset['observations'][row, 1] = 1+ dataset['observations'][row, 1] + (row/int(observation_size)) * 0.1
        #dataset['observations'][row, 1] = random.uniform(1, 2) # random.random()  s  
     
    print("dataset == ",dataset)  
                 
    #for row in dataset['observations']:
        #print("row",row) 
    #    row[0] = random.random()
    #    row[1] = random.random()
    
    print("dataset",dataset)

    print("dataset size",dataset['observations'].shape[0])
  
    print("dataset size",dataset['observations'].shape)
 
    #print("dataset['observations']",dataset['observations'])
 
    #print("observation_size",observation_size)
    #print("actionsize",dataset['actions'].shape[0])

    #print("TQ EXIT")
    #import sys 
    #sys.exit() 
    
    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset
  
def sequence_dataset(env, preprocess_fn):  ### Turn it into paths at differnt goals 
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)                     ### ACTUALLY GET THE DATA
    dataset = preprocess_fn(dataset)               ### Pre Process               

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]    
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1) ## If NO timeouts just cut off at the MAX episoide steps
 
        for k in dataset:                    ## loops through action, goals etc 
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])     ## Creating data_ array 

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}                         ## Create individual episodes data 
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)             ## More pre process
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
 

#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations']) 
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode

import csv

def write_dataset_to_csv(dataset, csv_file, perform=True):
    # Check if perform is True
    if perform:
        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['Actions', 'Observations', 'Rewards', 'Terminals', 'Timeouts'])

            # Write each row of data 
            for i in range(min(len(dataset['actions']), 1010000)):
                writer.writerow([
                    dataset['actions'][i],
                    dataset['observations'][i],
                    dataset['rewards'][i],
                    dataset['terminals'][i],
                    dataset['infos/goal'][i],
                    dataset['timeouts'][i]
                ])