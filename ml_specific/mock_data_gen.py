import numpy as np
import random
from noise_gen import add_jitter, apply_scaling

def generate_gesture_sequence(gesture_type="circle", num_samples=150):
    """Generates the isolated (N, 3) arrays for specific shapes."""
    t = np.linspace(0, 1, num_samples)
    ax, ay = np.zeros(num_samples), np.zeros(num_samples)
    az = np.full(num_samples, 9.8) 

    if gesture_type == "circle":
        ax = -2.0 * np.cos(2 * np.pi * t)
        ay = -2.0 * np.sin(2 * np.pi * t)
    elif gesture_type == "square":
        segments = np.array_split(np.arange(num_samples), 4)
        ax[segments[0]] = np.sin(np.linspace(0, 2 * np.pi, len(segments[0]))) * 2.0
        ay[segments[1]] = np.sin(np.linspace(0, 2 * np.pi, len(segments[1]))) * 2.0
        ax[segments[2]] = -np.sin(np.linspace(0, 2 * np.pi, len(segments[2]))) * 2.0
        ay[segments[3]] = -np.sin(np.linspace(0, 2 * np.pi, len(segments[3]))) * 2.0
    elif gesture_type == "triangle":
        segments = np.array_split(np.arange(num_samples), 3)
        pulse1 = np.sin(np.linspace(0, 2 * np.pi, len(segments[0]))) * 2.0
        ax[segments[0]], ay[segments[0]] = pulse1, pulse1
        pulse2 = np.sin(np.linspace(0, 2 * np.pi, len(segments[1]))) * 2.0
        ax[segments[1]], ay[segments[1]] = pulse2, -pulse2
        ax[segments[2]] = -np.sin(np.linspace(0, 2 * np.pi, len(segments[2]))) * 3.0

    # Add baseline noise to the movement
    ax += np.random.randn(num_samples) * 0.05
    ay += np.random.randn(num_samples) * 0.05
    az += np.random.randn(num_samples) * 0.05

    return np.column_stack((ax, ay, az))

def continuous_stream_simulator(total_samples=2000):
    stream = []
    samples_generated = 0

    while samples_generated < total_samples:
        # 1. Idle time (User holding the device still)
        idle_length = random.randint(100, 300)
        for _ in range(idle_length):
            stream.append([np.random.randn()*0.05, np.random.randn()*0.05, 9.8 + np.random.randn()*0.05])
            samples_generated += 1
            if samples_generated >= total_samples: break
            
        if samples_generated >= total_samples: break

        # 2. Inject a random gesture (User draws a shape)
        gesture_name = random.choice(["circle", "square", "triangle"])
        print(f"--- [Simulator] User starts drawing a {gesture_name.upper()} at sample {samples_generated} ---")
        gesture_data = generate_gesture_sequence(gesture_name, num_samples=150)

        # 3. Apply noise using the noise_gen.py file
        gesture_data = add_jitter(gesture_data, (random.random() * 0.2) + 0.5)
        gesture_data = apply_scaling(gesture_data, (random.random() * 0.2) + 0.6, (random.random() * 0.2) + 1.2)

        for sample in gesture_data:
            stream.append(sample)
            samples_generated += 1
            if samples_generated >= total_samples: break

    return np.array(stream)[:total_samples]