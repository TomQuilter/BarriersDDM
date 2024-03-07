for row in range(10):    ## int(observation_size * 0.9) // 10
        pair_index = row % 5
        print(row, " = row", pair_index)

class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

def pet_factory(pet_type):
    if pet_type == "dog":
        return Dog()
    elif pet_type == "cat":
        return Cat()
    else:
        raise ValueError("Unknown pet type")

# Usage
pet = pet_factory("dog")
print(pet.speak())  # Outputs: Woof!

pet = pet_factory("catt")
print(pet.speak())  # Outputs: Meow!
 
 

 
import torch

# Given trajectory points
trajectory = {
    0: torch.tensor([[-0.7, -0.7, 1.0000, -0.8000]]),      ## Blue dot
    64: torch.tensor([[-0.7, 0.9, -1.0000, -0.8000]]),      ## green Dot
    127: torch.tensor([[0.7, 0.9, 1.0000, -0.8000]])      ## Red Dot
}

# Calculate step sizes for interpolation
x_step_1 = (trajectory[64][0][0] - trajectory[0][0][0]) / 64
y_step_1 = (trajectory[64][0][1] - trajectory[0][0][1]) / 64
x_step_2 = (trajectory[127][0][0] - trajectory[64][0][0]) / 63
y_step_2 = (trajectory[127][0][1] - trajectory[64][0][1]) / 63

# Interpolate for indices 1 to 63
for i in range(1, 64):
    x = trajectory[0][0][0] + i * x_step_1
    y = trajectory[0][0][1] + i * y_step_1
    trajectory[i] = torch.tensor([[x, y, 1.0000, -0.8000]])

# Interpolate for indices 65 to 126
for i in range(65, 127):
    x = trajectory[64][0][0] + (i - 64) * x_step_2
    y = trajectory[64][0][1] + (i - 64) * y_step_2
    trajectory[i] = torch.tensor([[x, y, -1.0000, -0.8000]])
 
print("Trajectory = ", trajectory)  