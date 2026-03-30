# filename: imu_pipeline.py

import numpy as np
from collections import deque

# -----------------------------
# CONFIGURATION
# -----------------------------
WINDOW_SIZE = 200    # number of samples per window
STEP_SIZE = 50       # step to slide window forward (overlap if STEP_SIZE < WINDOW_SIZE)
CHANNELS = 3         # ax, ay, az
COMPUTE_MAGNITUDE = True  # whether to compute magnitude channel

# -----------------------------
# SIMULATED IMU STREAM FUNCTION
# -----------------------------
def generate_fake_imu_sample():
    """
    Simulate one accelerometer sample (ax, ay, az)
    Replace this function with STM32 serial read in the future
    """
    ax = np.random.randn() * 0.05
    ay = np.random.randn() * 0.05
    az = 9.8 + np.random.randn() * 0.05
    return np.array([ax, ay, az])

# -----------------------------
# SLIDING WINDOW PIPELINE
# -----------------------------
class SlidingWindowPipeline:
    def __init__(self, window_size=WINDOW_SIZE, step_size=STEP_SIZE, channels=CHANNELS):
        self.window_size = window_size
        self.step_size = step_size
        self.channels = channels
        self.buffer = deque(maxlen=window_size)
        self.sample_count = 0

    def add_sample(self, sample):
        """
        Add a single accelerometer sample to the buffer.
        Returns normalized window if a new step is reached, else None.
        """
        self.buffer.append(sample)
        self.sample_count += 1

        if len(self.buffer) < self.window_size:
            return None  # buffer not full yet

        # Only produce a window every step_size samples
        if (self.sample_count - self.window_size) % self.step_size == 0:
            window = np.array(self.buffer)
            # Normalize per channel
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0) + 1e-6
            window_norm = (window - mean) / std

            # Optional magnitude channel
            if COMPUTE_MAGNITUDE:
                mag = np.sqrt(np.sum(window_norm**2, axis=1, keepdims=True))
                window_norm = np.hstack([window_norm, mag])  # shape: (window_size, channels+1)

            return window_norm
        else:
            return None

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    pipeline = SlidingWindowPipeline()

    # Simulate continuous stream of 1000 samples
    for i in range(1000):
        sample = generate_fake_imu_sample()
        window = pipeline.add_sample(sample)
        if window is not None:
            print(f"Window ready for inference: shape {window.shape}")
            # Here you would feed `window` to your model
