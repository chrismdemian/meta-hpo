import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split # To split the dataset into train and test randomly
from sklearn.preprocessing import StandardScaler # Normalization
from config import DATA_PATH, META_PATH, TEST_SIZE, SEED
import json

def load_data(path=DATA_PATH):
    
    df = pd.read_csv(path)

    # Opens json file and loads into a dict
    with open(META_PATH) as f:
        meta = json.load(f)

    # Match dataset to features
    meta_lookup = {}
    for name, desc in zip(meta['dataset_names'], meta['dataset_descriptors']):
        # Get first 2 features as image size and layers are the same for all
        meta_lookup[name] = desc[:2]

    # Gets list of all 30 one-hot column names
    dataset_cols = [col for col in df.columns if 'cat__dataset' in col]

    # Finds which dataset column is a 1, gets name and looks up features
    def get_meta_features(row):
        for col in dataset_cols:
            if row[col] == 1.0:
                # Strips prefix to get name
                name = col.replace('cat__dataset_mtlbm/', '')
                # Return [0,0] if name isn't found
                return meta_lookup.get(name, [0,0])
        return [0,0]
    
    # Run function row by row and split into 2 columns
    meta_features = df.apply(get_meta_features, axis=1, result_type='expand')
    meta_features.columns = ['num_images', 'num_classes']

    # Remove all 30 old one hot dataset columns and add the 2 new meaninful ones
    df = df.drop(columns=dataset_cols)
    df['num_images'] = meta_features['num_images']
    df['num_classes'] = meta_features['num_classes']

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