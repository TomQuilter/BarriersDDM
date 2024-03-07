import torch

x = torch.tensor([[[-0.8490, -0.7648, -1.6678, -0.6449]]], device='cuda:0')
t = 0  # Assuming 't' points to the correct time step in your data
action_dim = 0  # Assuming this is the correct starting index for action dimensions in your data
val_clone = torch.tensor([[-0.9306, -0.9298, -1.0000, -1.0000]], device='cuda:0')

# The operation to correctly update only the first two elements of x[:, t, action_dim:]
# and keep the remaining elements unchanged.
x[:, t, 0:2] = val_clone[:, :2]
 
# Demonstrating the result to ensure the last two numbers are preserved.
print(x)


def generate_conditions(num_rows):
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

# Example usage
# conditions = generate_conditions(128)
# This will generate conditions for 128 rows, but you can replace 128 with any desired number of rows.
   
 
conditions = generate_conditions(3)
   
# print("conditions", conditions) 