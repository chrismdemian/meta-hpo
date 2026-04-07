import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_unified_hpob_datasets
from model import UnifiedHPOModel
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, LR_STEP_SIZE, LR_GAMMA, WEIGHT_DECAY

# Get the datasets and model info
train_dataset, test_dataset, ss_info, n_datasets = get_unified_hpob_datasets()

# Load them
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnifiedHPOModel(ss_info, n_datasets).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

print(f"Training unified model on {len(train_dataset)} samples across {len(ss_info)} search spaces")
print(f"Device: {device}")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch, ss_batch in train_loader:
        X_batch, y_batch, ss_batch = X_batch.to(device), y_batch.to(device), ss_batch.to(device)

        predictions = model(X_batch, ss_batch)
        loss = criterion(predictions.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    scheduler.step()

torch.save(model.state_dict(), 'checkpoints/unified_model.pth')
print(f"Model saved to checkpoints/unified_model.pth")
