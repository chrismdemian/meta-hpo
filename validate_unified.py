import torch
import numpy as np
from dataset import get_unified_hpob_datasets
from model import UnifiedHPOModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data and model
_, test_dataset, ss_info, n_datasets = get_unified_hpob_datasets()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnifiedHPOModel(ss_info, n_datasets).to(device)
model.load_state_dict(torch.load('checkpoints/unified_model.pth', map_location=device))
model.eval()

# Get predictions
with torch.no_grad():
    X_test = test_dataset.X.to(device)
    y_test = test_dataset.y.numpy()
    ss_test = test_dataset.ss_idx.to(device)
    predictions = model(X_test, ss_test).squeeze().cpu().numpy()

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"R2:   {r2:.4f}")

print("\nSample Predictions:")
indices = np.random.RandomState(42).choice(len(y_test), 15, replace=False)
for i in indices:
    print(f"  Actual: {y_test[i]:.4f}, Predicted: {predictions[i]:.4f}, Error: {abs(y_test[i] - predictions[i]):.4f}")
