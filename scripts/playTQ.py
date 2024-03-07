import torch

import minari

import torch

import numpy as np
import random


x = random.uniform(0, 1)
print("x= ", x)
# Determine the range for y based on the value of x
if 0 <= x <= 0.2:
    y_range_start, y_range_end = 0, 1
elif 0.2 < x <= 0.4:
    y_range_start, y_range_end = 0, 0.2
elif 0.4 < x <= 0.6:
    y_range_start, y_range_end = 0, 0.6
elif 0.6 < x <= 0.8:
    y_range_start, y_range_end = 0, 0.2
elif 0.8 < x <= 1:  # This condition is supposed to be always true if x is between 0 and 0.8
    y_range_start, y_range_end = 0, 1
else:
    # If x is not in the range [0, 0.8], we don't draw y
    y_range_start, y_range_end = None, None

# If a valid range for y exists, draw y randomly from that range
if y_range_start is not None and y_range_end is not None:
    y = random.uniform(y_range_start, y_range_end)
else:
    y = None

print("y= ", y)

 




# Function to replace the first two numbers of each row in the array within the dictionary
def replace_first_two_with_random_in_dict(data_dict, key):
    # Check if the key exists in the dictionary
    if key in data_dict:
        # Get the array from the dictionary
        array = data_dict[key]
        # Replace the first two numbers of each row with random floats
        for row in array:
            row[0] = random.random()
            row[1] = random.random()
    else:
        print(f"Key '{key}' not found in the dictionary.")

# Example dictionary with 'observations' key
data_dict = {
    'observations': np.array([
        [0.46749386, 0.14776532, 0.00981035, 0.02174424],
        [0.46749386, 0.14776532, -0.12562364, -0.04433781],
        [0.46749386, 0.14776532, -0.3634883, 0.11453988],
        [0.46749386, 0.14776532, -4.484303, 0.09555068],
        [0.46749386, 0.14776532, -4.4510083, 0.06509537],
        [0.46749386, 0.14776532, -4.202244, 0.05324839]
    ])
}

# Replace the first two numbers with random values
replace_first_two_with_random_in_dict(data_dict, 'observations')

# Check the modified array
print(data_dict['observations'])


# Your provided 'cond' dictionary
cond = {
    0: torch.tensor([[-0.5100,  0.0400,  0.0019,  0.0042]], device='cuda:0'),
    127: torch.tensor([[ 0.6872,  0.8385, -0.7158,  0.0234]], device='cuda:0')
}

print("cond orig = ", cond)

# New dictionary to store the modified tensors
new_cond = {}

# Iterate over the original dictionary and create new tensors
for key, tensor in cond.items():
    num_columns = tensor.shape[1]
    new_tensor = torch.zeros(10, num_columns, device='cuda:0')
    new_tensor[:, 0] = 0.1
    if num_columns > 1:
        new_tensor[:, 1] = 0.88
    new_cond[key] = new_tensor

# Output the new dictionary
print(new_cond)





#dataset = minari.load_dataset('minigrid-fourrooms-v0', download=True)
#env  = dataset.recover_environment()
#eval_env = dataset.recover_environment(eval_env=True)

#assert env.spec == eval_env.spec

tensor = torch.tensor([1, 2, 3])
print(tensor.device)  # This will print 'cpu' if the tensor is on the CPU



def my_function(arg1, arg2, arg3):
    print(arg1)
    print(arg2)
    print(arg3)

my_list = [1, 2, 3]
print("my_list",my_list)
print("my_list",*my_list)
#my_function(my_list)