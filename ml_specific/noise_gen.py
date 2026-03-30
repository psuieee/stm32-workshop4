import numpy as np

def add_jitter(gesture_data, noise_level=0.1):
    """
    Simulates a 'shaky hand' or standard sensor noise.
    Adds random Gaussian noise to every point in the sequence.
    """
    # Create an array of random noise the exact same shape as our data
    noise = np.random.normal(loc=0.0, scale=noise_level, size=gesture_data.shape)
    return gesture_data + noise

def apply_scaling(gesture_data, min_scale=0.8, max_scale=1.2):
    """
    Simulates a user making a larger or smaller version of the gesture.
    Multiplies the entire sequence by a single random factor.
    """
    # Pick a random number between the min and max
    scale_factor = np.random.uniform(min_scale, max_scale)
    return gesture_data * scale_factor

def apply_time_shift(gesture_data, max_shift=15):
    """
    Simulates the user starting the gesture slightly early or late 
    after hitting the 'Record' button.
    """
    # Pick a random number of frames to shift forward or backward
    shift_amount = np.random.randint(-max_shift, max_shift)
    
    # np.roll pushes the data forward/backward. 
    # What falls off the edge wraps around to the other side.
    shifted_data = np.roll(gesture_data, shift_amount, axis=0)
    
    return shifted_data