import torch

import minari

import torch

import numpy as np
import random
 
random.seed(42) 

class Config:
    def __init__(self, config_fn, **kwargs):
        self.config_fn = config_fn
        self.kwargs = kwargs

    def __call__(self):
        # Assume config_fn is a function that creates and configures an object
        return self.config_fn(**self.kwargs)

# Example function to be used with Config
def create_model(horizon, transition_dim):
    # Dummy function to simulate model creation
    print(f"Creating model with horizon={horizon} and transition_dim={transition_dim}")
    return "ModelInstance"

# Using Config to store configuration and then create a model
utils = {'Config': Config}  # Simulating the utils module
model_config = utils['Config'](create_model, horizon=10, transition_dim=5)

# Later, when we want to instantiate the model
model_instance = model_config()  # Calls create_model(**kwargs) under the hood
print(model_instance)
 
dataset = {
    'observations': [ 
        [1, 1],  # Point 1
        [3, 4],  # Point 2
    ]
}
  
div = (288 // 128) % 2   ## ( 8 // 5) // 2
 
# Rotation function
def rotate_around_point(x, y, center_x=0.5, center_y=0.5):
    translated_x = x - center_x
    translated_y = y - center_y 
    rotated_x = translated_y
    rotated_y = -translated_x
    new_x = rotated_x + center_x
    new_y = rotated_y + center_y
    return new_x, new_y

 
import matplotlib.pyplot as plt
import numpy as np
 
# Assuming coordinates_grid is already defined as shown previously
# If not, you would need to regenerate it using:
# X_equally_spaced_points = np.linspace(0, 1, 10)
# Y_equally_spaced_points = np.linspace(0, 0.2, 10)
# X, Y = np.meshgrid(X_equally_spaced_points, Y_equally_spaced_points)
# coordinates_grid = np.column_stack((X.ravel(), Y.ravel()))

def generate_all_coords(CalibrationValue):
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
    
    Y_equally_spaced_points4 = np.linspace(0.2, 0.6, 5) - CalibrationValue
    X_equally_spaced_points4 = np.linspace(0.4, 0.6, 3) - CalibrationValue
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
    
    return All_coords
    
calibration_value = 0
 
# Generate all coordinates based on the provided calibration value
All_coords = generate_all_coords(calibration_value)
 
# Print the generated coordinates
print(All_coords)

 
    
# All_coords = np.array([rotate_around_point(x, y) for x, y in All_coords])
     
# Plotting the coordinates
plt.figure(figsize=(8, 4))
plt.scatter(All_coords[:, 0], All_coords[:, 1], color='blue', label='Coordinates')
  
# Setting plot labels and title
plt.title('Graph Showing All Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
#plt.legend() 

# Show the plot with grid
plt.grid(True)
plt.show()
 

 
 



 


def custom_random_uniform(low1, high1, low2, high2):
        # Decide which range to use
        if random.random() < 0.5:  # 50% chance for each range
            return random.uniform(low1, high1)
        else:
            return random.uniform(low2, high2)
        
dataset = {'observations': np.zeros((10000, 4))}

# Initialize an empty list to store the pairs
pairs = []

for LoopNumber in range(256):
    RandomNumber = random.uniform(0.1, 1)
    observation = RandomNumber - 0.5  # Adjusted RandomNumber for observation

    # Determine y based on the value of RandomNumber
    if 0 <= RandomNumber <= 0.2:
        y = random.uniform(0, 1)
    elif 0.2 < RandomNumber <= 0.4:
        y = custom_random_uniform(0, 0.2, 0.8, 1)
    elif 0.4 < RandomNumber <= 0.6:
        y = custom_random_uniform(0, 0.4, 0.9, 1)
    elif 0.6 < RandomNumber <= 0.8:
        y = custom_random_uniform(0, 0.2, 0.8, 1)
    elif 0.8 < RandomNumber <= 1:
        y = random.uniform(0, 1)
    else:
        y = 0  # This case should never be hit due to the range of RandomNumber

    # Store the pair (observation, y)
    pairs.append((observation, y))

# Print out the stored pairs
for index, (observation, y) in enumerate(pairs):
    print(f"Pair {index + 1}: Observation = {observation}, y = {y}")


for row in range(300):
    print("row, ",row) 
    pair_index = row % 25  # This cycles through 0-255 for each row
    print("pair_index", pair_index)
    observation, y = pairs[pair_index]
    
    # Assign the values to the dataset
    dataset['observations'][row, 2] = observation
    dataset['observations'][row, 3] = y

# To verify, let's print out some of the dataset values
# Print the first 10 and last 10 rows for a quick check
print("First 10 rows:")
for i in range(270):
    print(dataset['observations'][i])

print("\nLast 10 rows:")
for i in range(9990, 10000):
    print(dataset['observations'][i])



    #### Random Array

    def custom_random_uniform(low1, high1, low2, high2):
        # Decide which range to use
        if random.random() < 0.5:  # 50% chance for each range
            return random.uniform(low1, high1)
        else:
            return random.uniform(low2, high2)
        
    # Initialize an empty list to store the pairs
    pairs = []

    for NumberofBarrierObservations in range(1023):  
        RandomNumber = random.uniform(0, 1)  ## Select the Random X Barrier Co-ordinate, Could do Linespacing too

        # Determine y based on the value of RandomNumber
        if 0 <= RandomNumber <= 0.2:
            y = random.uniform(0, 1)
        elif 0.2 < RandomNumber <= 0.4:
            y = custom_random_uniform(0, 0.2, 0.8, 1)
        elif 0.4 < RandomNumber <= 0.6:
            y = custom_random_uniform(0, 0.6, 0.8, 1)
        elif 0.6 < RandomNumber <= 0.8: 
            y = custom_random_uniform(0, 0.2, 0.8, 1)
        elif 0.8 < RandomNumber <= 1:
            y = random.uniform(0, 1)
        else:
            y = 0  # This case should never be hit due to the range of RandomNumber

        # Store the pair (observation, y)


        ##Calibrate
        BarrierXCoordinate = RandomNumber - 0.5  # Adjusted RandomNumber for observation
        BarrierYCoordinate = y - 0.5  # Adjusted RandomNumber for observation

        pairs.append((BarrierXCoordinate, BarrierYCoordinate))