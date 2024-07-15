import math

def calculate_distance(start, end):
    
    return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    #return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)


def cut_tape(current_position, target_position, d):
    
    #Calculate the cutting point for the tape cutter to ensure the tape ends exactly at the target position.
    remaining_distance = calculate_distance(current_position, target_position)
    
    # Distance to cut the tape from the current position
    cutting_distance = remaining_distance - d
    
    # Ensure the cutting distance is not negative
    if cutting_distance < 0:
        cutting_distance = 0
    
    return cutting_distance

# Example usage:
current_position = (5, 5)
target_position = (10, 10)
d = 2

cut_distance = cut_tape(current_position, target_position, d)
print(f"The tape should be cut after traveling {cut_distance} units from the current position.")
