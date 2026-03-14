import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_datasets
from model import HPOModel
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS

# Get the datasets
train_dataset, test_dataset = get_datasets()

# Load them
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HPOModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Make a prediction
        predictions = model(X_batch)
        
        # Calculate loss
        loss = criterion(predictions.squeeze(), y_batch)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'checkpoints/model.pth')
print(f"Model saved to checkpoints/model.pth")