import torch
import numpy as np
from dataset import get_hpob_datasets
from model import HPOModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data and model (no need for training data)
_, test_dataset = get_hpob_datasets()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HPOModel().to(device)
model.load_state_dict(torch.load('checkpoints/model.pth', map_location=device))
model.eval()

# Get predictions for the test set
with torch.no_grad():
    X_test = test_dataset.X.to(device)
    y_test = test_dataset.y.numpy()
    predictions = model(X_test).squeeze().cpu().numpy()

# Calculate and output metrics

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE:  {np.sqrt(mse):.4f}")
print(f"R2:  {r2:.4f}")

# Print some sample predictions
print("\nSample Predictions:")
for i in range(10):
    print(f"  Actual: {y_test[i]:6.2f}, Predicted: {predictions[i]:6.2f}, Error: {abs(y_test[i] - predictions[i]):6.2f}")
