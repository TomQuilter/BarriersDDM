keys = [0, 64, self.horizon - 1] 

# Use a dictionary comprehension to construct the dictionary dynamically.
# This approach avoids the direct use of `eval` for better safety and readability.
conditions = {k: observations[min(k, len(observations) - 1)] for k in keys}


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


result_string = generate_string(5)
print(result_string)
