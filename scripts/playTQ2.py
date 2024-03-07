import pickle

# Replace 'your_file.pkl' with your pickle file path
file_path = 'dataset_config.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the content of the file into a Python object
    data = pickle.load(file)

# Now you can use the 'data' object as a normal Python object
print(data)
