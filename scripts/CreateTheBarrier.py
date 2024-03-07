import torch

import minari

import torch

import numpy as np
import random

random.seed(42)  

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