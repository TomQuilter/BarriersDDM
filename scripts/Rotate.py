import numpy as np

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

# Example usage
px, py = 5, 5  # Point to rotate
angle_degrees = 180  # Rotation angle in degrees
cx, cy = 0, 0  # Center of rotation
 
rotated_x, rotated_y = rotate_point(px, py, angle_degrees, cx, cy)
 
# Print the separated coordinates
print(f"Rotated X: {rotated_x}")
print(f"Rotated Y: {rotated_y}")