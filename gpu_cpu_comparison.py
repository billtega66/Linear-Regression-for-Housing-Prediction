import pandas as pd
import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('housing[1].csv')
data['ocean_proximity'] = data['ocean_proximity'].replace({'NEAR BAY': 1, 'INLAND': 0, '<1H OCEAN': 2, 'NEAR OCEAN': 3, 'ISLAND': 4})
data['ocean_proximity'] = data['ocean_proximity'].astype(float)
data = data.dropna()

# Prepare features and target
X = data[['median_income', 'households']]
y = data['median_house_value']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to measure execution time
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Function to convert data to PyTorch tensors
def prepare_data_for_torch(X, y):
    X_tensor = torch.FloatTensor(X.values).to(device)
    y_tensor = torch.FloatTensor(y.values).to(device)
    return X_tensor, y_tensor

# GPU Linear Regression implementation
class GPULinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = torch.zeros(n_features, 1, device=device, requires_grad=True)
        self.bias = torch.zeros(1, device=device, requires_grad=True)
        
        for _ in range(epochs):
            y_pred = torch.matmul(X, self.weights) + self.bias
            loss = torch.mean((y_pred - y.reshape(-1, 1)) ** 2)
            
            loss.backward()
            
            with torch.no_grad():
                self.weights -= learning_rate * self.weights.grad
                self.bias -= learning_rate * self.bias.grad
                
                self.weights.grad.zero_()
                self.bias.grad.zero_()
    
    def predict(self, X):
        return torch.matmul(X, self.weights) + self.bias

# Prepare data for GPU
print("Preparing data for GPU...")
X_train_gpu, y_train_gpu = prepare_data_for_torch(X_train, y_train)
X_test_gpu, y_test_gpu = prepare_data_for_torch(X_test, y_test)

# Initialize models
gpu_model = GPULinearRegression()
cpu_model = LinearRegression()

# Train and measure time for GPU model
print("Training GPU model...")
gpu_result, gpu_time = measure_time(gpu_model.fit, X_train_gpu, y_train_gpu)
print(f"GPU Training time: {gpu_time:.4f} seconds")

# Train and measure time for CPU model
print("Training CPU model...")
cpu_result, cpu_time = measure_time(cpu_model.fit, X_train, y_train)
print(f"CPU Training time: {cpu_time:.4f} seconds")

# Make predictions
print("Making predictions...")
gpu_predictions = gpu_model.predict(X_test_gpu).cpu().detach().numpy()
cpu_predictions = cpu_model.predict(X_test)

# Calculate metrics
gpu_mse = mean_squared_error(y_test, gpu_predictions)
cpu_mse = mean_squared_error(y_test, cpu_predictions)

gpu_r2 = r2_score(y_test, gpu_predictions)
cpu_r2 = r2_score(y_test, cpu_predictions)

print(f"\nGPU Model MSE: {gpu_mse:.2f}, R2 Score: {gpu_r2:.4f}")
print(f"CPU Model MSE: {cpu_mse:.2f}, R2 Score: {cpu_r2:.4f}")

# Visualize the performance comparison
times = [gpu_time, cpu_time]
labels = ['GPU', 'CPU']

plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=['blue', 'red'])
plt.title('Training Time Comparison: GPU vs CPU')
plt.ylabel('Time (seconds)')
plt.xlabel('Device')
plt.savefig('gpu_cpu_comparison.png')
plt.show() 