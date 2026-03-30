import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from mock_data_gen import generate_gesture_sequence
from noise_gen import add_jitter, apply_scaling, apply_time_shift
from data_preprocessing import SlidingWindowPipeline

class GestureCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=3):
        super(GestureCNN, self).__init__()

        self.net = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
                                 nn.LazyConv1d(out_channels=8, kernel_size=3), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
                                 nn.Flatten(),
                                 nn.LazyLinear(64), nn.ReLU(), nn.Dropout(0.5),
                                 nn.LazyLinear(num_classes))

    def forward(self, x):
        return self.net(x)

def create_synthetic_dataset(samples_per_class=200, window_size=150):
    pipeline = SlidingWindowPipeline(window_size=window_size, step_size=window_size)
    xs, ys = [], []

    gesture_names = ["circle", "square", "triangle"]

    for label, gesture in enumerate(gesture_names):
        for i in range(samples_per_class):
            gesture_data = generate_gesture_sequence(gesture_type=gesture, num_samples=window_size)
            gesture_data = add_jitter(gesture_data, noise_level=0.12)
            gesture_data = apply_scaling(gesture_data, min_scale=0.8, max_scale=1.2)
            gesture_data = apply_time_shift(gesture_data, max_shift=20)

            pipeline.buffer.clear()
            pipeline.sample_count = 0

            for sample in gesture_data:
                window = pipeline.add_sample(sample)
                if window is not None:
                    xs.append(window)
                    ys.append(label)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    # Shuffle dataset
    perm = np.random.permutation(len(ys))
    xs, ys = xs[perm], ys[perm]

    return xs, ys

if __name__ == "__main__":
    window_size = 150
    samples_per_class = 300

    print("[DATA] Generating synthetic dataset...")
    X, y = create_synthetic_dataset(samples_per_class=samples_per_class, window_size=window_size)

    # --- THE CRITICAL PYTORCH FIX ---
    # PyTorch wants (Batch, Channels, Time), not (Batch, Time, Channels)
    # So we transpose dimensions 1 and 2
    X = np.transpose(X, (0, 2, 1)) 
    
    # Convert numpy arrays to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split data (80% Train, 20% Val)
    test_split = 0.2
    split = int((1 - test_split) * len(X))

    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]

    print(f"[DATA] PyTorch Tensors ready: X_train={X_train.shape}, y_train={y_train.shape}")

    # Create DataLoaders (Handles batching for us)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize Model, Loss Function, and Optimizer
    model = GestureCNN(in_channels=4, num_classes=3)
    criterion = nn.CrossEntropyLoss() # This handles both the Softmax and the loss math
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. The Training Loop
    epochs = 15
    print("\n[TRAIN] Starting training...")
    
    for epoch in range(epochs):
        model.train() # Set to training mode (enables dropout)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()           # Clear old gradients
            outputs = model(batch_X)        # Forward pass
            loss = criterion(outputs, batch_y) # Calculate error
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        # Validation step
        model.eval() # Set to evaluation mode (disables dropout)
        correct_val = 0
        total_val = 0
        
        # Don't track gradients during validation (saves memory/time)
        with torch.no_grad(): 
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        # Print epoch stats
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    # Save the model
    model_save_path = "gesture_cnn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[TRAIN] Training complete. Model weights saved to {model_save_path}")