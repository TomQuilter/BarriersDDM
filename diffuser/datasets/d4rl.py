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

def generate_all_coords(CalibrationValue,Orientation):
    X_equally_spaced_points = np.linspace(0, 1, 11) - CalibrationValue
    Y_equally_spaced_points = np.linspace(0, 0.2, 3) - CalibrationValue
    
    X, Y = np.meshgrid(X_equally_spaced_points, Y_equally_spaced_points)
    coordinates_grid = np.column_stack((X.ravel(), Y.ravel()))
    
    X_equally_spaced_points2 = np.linspace(0, 0.2, 3) - CalibrationValue
    Y_equally_spaced_points2 = np.linspace(0.3, 0.7, 5) - CalibrationValue
    X2, Y2 = np.meshgrid(X_equally_spaced_points2, Y_equally_spaced_points2)
    coordinates_grid2 = np.column_stack((X2.ravel(), Y2.ravel()))
    
    X_equally_spaced_points3 = np.linspace(0.8, 1, 3) - CalibrationValue
    Y_equally_spaced_points3 = np.linspace(0.2, 0.8, 7) - CalibrationValue
    X3, Y3 = np.meshgrid(X_equally_spaced_points3, Y_equally_spaced_points3)
    coordinates_grid3 = np.column_stack((X3.ravel(), Y3.ravel()))
  
    if(Orientation == "Left"): 
        Y_equally_spaced_points4 = np.linspace(0.2, 0.6, 5) - CalibrationValue
        X_equally_spaced_points4 = np.linspace(0.4, 0.6, 3) - CalibrationValue
        X4, Y4 = np.meshgrid(X_equally_spaced_points4, Y_equally_spaced_points4)
        coordinates_grid4 = np.column_stack((X4.ravel(), Y4.ravel()))

    if(Orientation == "Down"):  
        Y_equally_spaced_points4 = np.linspace(0.4, 0.6, 3) - CalibrationValue
        X_equally_spaced_points4 = np.linspace(0.2, 0.6, 5) - CalibrationValue
        X4, Y4 = np.meshgrid(X_equally_spaced_points4, Y_equally_spaced_points4)
        coordinates_grid4 = np.column_stack((X4.ravel(), Y4.ravel()))
  
    if(Orientation == "Right"): 
        Y_equally_spaced_points4 = np.linspace(0.4, 0.8, 5) - CalibrationValue
        X_equally_spaced_points4 = np.linspace(0.4, 0.6, 3) - CalibrationValue
        X4, Y4 = np.meshgrid(X_equally_spaced_points4, Y_equally_spaced_points4)
        coordinates_grid4 = np.column_stack((X4.ravel(), Y4.ravel())) 
 
    if(Orientation == "Up"):    
        Y_equally_spaced_points4 = np.linspace(0.4, 0.6, 3) - CalibrationValue
        X_equally_spaced_points4 = np.linspace(0.4, 0.8, 5) - CalibrationValue
        X4, Y4 = np.meshgrid(X_equally_spaced_points4, Y_equally_spaced_points4)
        coordinates_grid4 = np.column_stack((X4.ravel(), Y4.ravel())) 
     
    X_equally_spaced_points7 = np.linspace(0, 1, 11) - CalibrationValue
    Y_equally_spaced_points7 = np.linspace(0.8, 1, 3) - CalibrationValue
    X7, Y7 = np.meshgrid(X_equally_spaced_points7, Y_equally_spaced_points7)
    coordinates_grid7 = np.column_stack((X7.ravel(), Y7.ravel()))
    
    X_equally_spaced_points8 = np.linspace(0, 1, 11) - CalibrationValue
    Y_equally_spaced_points8 = np.linspace(0.8, 0.8, 1) - CalibrationValue
    X8, Y8 = np.meshgrid(X_equally_spaced_points8, Y_equally_spaced_points8)
    coordinates_grid8 = np.column_stack((X8.ravel(), Y8.ravel()))
    
    All_coords = np.vstack((coordinates_grid, coordinates_grid2, coordinates_grid3, coordinates_grid4, coordinates_grid7, coordinates_grid8))
      
    print("All_coords", All_coords.shape)
 
    return All_coords
 
def rotate_point(px, py, angle_degrees, cx, cy):
    """ 
    Rotate a point (px, py) around a given point (cx, cy) by a given angle in degrees.
     
    Parameters:
    - px, py: Coordinates of the point to rotate.
    - angle_degrees: The rotation angle in degrees.
    - cx, cy: Coordinates of the center of rotation.
    
    Returns:
    - The rotated coordinates as a tuple (nx, ny).
    """
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Translate point to origin
    px_translated = px - cx
    py_translated = py - cy
    
    # Perform rotation
    nx = px_translated * np.cos(angle_radians) - py_translated * np.sin(angle_radians)
    ny = px_translated * np.sin(angle_radians) + py_translated * np.cos(angle_radians)
    
    # Translate point back
    nx += cx
    ny += cy
    
    return (nx, ny)

 
def get_dataset(env):
    dataset = env.get_dataset()     ## Get the DATA From the gym environment

    import random
    random.seed(42)  
 
    # Call the function to write the dataset to the CSV file
    csv_file = 'outputWithGoalsLDS.csv'
    write_dataset_to_csv(dataset, csv_file, perform=False)
        
    print("dataset['observations'][0, 0]",dataset['observations'][0, 0])
    print("dataset['observations'][0, 0]",dataset['observations'][0, 1])
    print("dataset['observations'][0, 0]",dataset['observations'][1, 0])   
    print("dataset['observations'][0, 0]",dataset['observations'][1, 1])

    dataset['observations'][0, 0] = 0.44
    dataset['observations'][0, 1] = 0.44   
  
    #dataset['observations'][1, 0] = 0.23
    #dataset['observations'][1, 1] = 0.23 
  
    dataset['observations'][2, 0] = 0.33
    dataset['observations'][2, 1] = 0.33   

    dataset['observations'][127, 0] = 0.51
    dataset['observations'][127, 1] = 0.51   
    
    dataset['observations'][128, 0] = 0.51
    dataset['observations'][128, 1] = 0.51 
    #dataset['observations'][129, 0] = 0.44
    #dataset['observations'][129, 1] = 0.44    
   
    observation_size = dataset['observations'].shape[0]

    print("dataset == ",dataset)  
   
    print("dataset['observations'].shape[0]",dataset['observations'].shape[0])

    Left_coords = generate_all_coords(0.5,"Left")   #/2   
    Up_coords = generate_all_coords(0.5,"Up")      #/1)
    Right_coords = generate_all_coords(0.5,"Right")
    Down_coords = generate_all_coords(0.5,"Down")
 
    All_coords = np.vstack((Left_coords, Up_coords))

    print("All_coords[0]",All_coords[0])
              
    for row in range(int(observation_size)):    ## int(observation_size * 0.9) // 10
              
        ### INSERT THE BARRIER BY OVERWRITTING THE VELOCITY CO-ORDINATES IN OBSERVATIONS

        pair_index = row % 256  # This cycles through 0-255 for each row - will be the row number for row = 0 upto 255 then back to 0 after 255
        dataset['observations'][row, 2], dataset['observations'][row, 3] = All_coords[pair_index]
          
        #OriginalXCoords = dataset['observations'][row, 0]
        #dataset['observations'][row, 0] = dataset['observations'][row, 1]
        #dataset['observations'][row, 1] = OriginalXCoords  

        #dataset['observations'][row, 0] = dataset['observations'][row, 0] + 0.4
        #dataset['observations'][row, 1] = dataset['observations'][row, 1] - 0.4
     
    
        ### ROTATE THE TRAJECTORIES WITH THE BARRIERS ####

        if  0 < row < 2500001:  # Alternate between the first 128 rows - ie: between the batches    
            pair_index = row % 128   
            dataset['observations'][row, 2], dataset['observations'][row, 3] = Left_coords[pair_index] 
            1==1     
 
        #if  ((row) // 128) % 2 != 0:  # Alternate between the first 128 rows - ie: between the batches
        if  250000 < row < 500001:  # Alternate between the first 128 rows - ie: between the batches
            pair_index = row % 128   
            dataset['observations'][row, 2], dataset['observations'][row, 3] = Up_coords[pair_index]  
              
            dataset['observations'][row, 0], dataset['observations'][row, 1] = rotate_point(dataset['observations'][row, 0], dataset['observations'][row, 1], 90, 2, 2)
 
            OriginalXCoords = dataset['observations'][row, 0]
            #dataset['observations'][row, 0] = dataset['observations'][row, 1]
            #dataset['observations'][row, 1] = OriginalXCoords

        if  500000 < row < 750001:  # Alternate between the first 128 rows - ie: between the batches
            pair_index = row % 128   
            dataset['observations'][row, 2], dataset['observations'][row, 3] = Right_coords[pair_index]  
             
            dataset['observations'][row, 0], dataset['observations'][row, 1] = rotate_point(dataset['observations'][row, 0], dataset['observations'][row, 1], 180, 2, 2)
 

        if  750000 < row < 1000000:  # Alternate between the first 128 rows - ie: between the batches
            pair_index = row % 128    
            dataset['observations'][row, 2], dataset['observations'][row, 3] = Down_coords[pair_index]  
              
            dataset['observations'][row, 0], dataset['observations'][row, 1] = rotate_point(dataset['observations'][row, 0], dataset['observations'][row, 1], 270, 2, 2)              
  
            1==1       

            #x = dataset['observations'][row, 0]  # Extract the original x, y coordinates
            #y = dataset['observations'][row, 1] 
            #new_x, new_y = rotate_around_point(x, y)  # Rotate the coordinate
            #dataset['observations'][row, 0] = new_x
            #dataset['observations'][row, 1] = new_y 
      
    print("dataset == ",dataset)  
                  
    
    print("dataset",dataset)

    print("dataset size",dataset['observations'].shape[0])
  
    print("dataset size",dataset['observations'].shape)

    max_value = max(dataset['observations'][0])
    
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
    # timeouts field. Keep old method for backwards compatability.  Is the (1,1) goal the OLD method??
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):   ### Loop One Million times Through every row of the dataset And ... 
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
 
    1==1
 

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