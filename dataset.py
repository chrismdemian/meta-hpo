import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split # To split the dataset into train and test randomly
from sklearn.preprocessing import StandardScaler # Normalization
from config import DATA_PATH, TEST_SIZE, SEED

def load_data(path=DATA_PATH):
    
    df = pd.read_csv(path)

    # Remove columns we don't need
    df = df.drop(columns=['Unnamed: 0', 'max_eval_top5', 'distributed', 'prefetcher', 'rank', 'world_size'])

    X = df.drop(columns=['max_eval_top1']) # Everything the model gets to see (hyperparameters etc)
    y = df['max_eval_top1'] # Accuracy we want it to predict

    # Split into training and 20% test with reproducible results (seed = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Normalize training data
    X_test = scaler.transform(X_test) # Normalize testing data from training normalization

    return X_train, X_test, y_train, y_test

class HPODataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   

def get_datasets(path=DATA_PATH):
    # Function to be used by other files to get the datasets
    
    X_train, X_test, y_train, y_test = load_data(path)

    train_dataset = HPODataset(X_train, y_train)
    test_dataset = HPODataset(X_test, y_test)

    return train_dataset, test_dataset