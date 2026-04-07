import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT

# Build a class for the neural network
class HPOModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 2 hidden layers with dropout and relu activation
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )

    # Forward method
    def forward(self, x):
        return self.network(x)

# Unified model with per-search-space input layers
EMBED_SIZE = 32

class UnifiedHPOModel(nn.Module):
    def __init__(self, ss_info, n_datasets):
        super().__init__()

        # Each search space gets its own input projection layer
        self.input_layers = nn.ModuleDict()
        for ss_idx, n_feat in ss_info.items():
            self.input_layers[str(ss_idx)] = nn.Linear(n_feat, EMBED_SIZE)

        # Shared hidden layers: embedding + dataset one-hot -> prediction
        shared_input = EMBED_SIZE + n_datasets
        self.shared = nn.Sequential(
            nn.Linear(shared_input, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )

    def forward(self, x, ss_idx):
        # x contains padded hyperparams (first 18) + dataset one-hot (rest)
        hp_features = x[:, :18]
        ds_features = x[:, 18:]

        # Process each sample through its search space's input layer
        embeddings = torch.zeros(x.size(0), EMBED_SIZE, device=x.device)
        for ss_id in ss_idx.unique():
            mask = ss_idx == ss_id
            n_feat = self.input_layers[str(ss_id.item())].in_features
            embeddings[mask] = self.input_layers[str(ss_id.item())](hp_features[mask, :n_feat])

        # Concatenate embedding with dataset one-hot and pass through shared layers
        combined = torch.cat([embeddings, ds_features], dim=1)
        return self.shared(combined)