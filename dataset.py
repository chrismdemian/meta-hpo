import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split # To split the dataset into train and test randomly
from sklearn.preprocessing import StandardScaler # Normalization
from config import DATA_PATH, META_PATH, TEST_SIZE, SEED
import json
import numpy as np

# Load all data for just the micro dataset
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

# Load all data for micro, mini, and extended datasets
def load_all_data():
    
    # Load all three datasets
    df_micro = pd.read_csv('hpo-data/micro/micro/args_table.csv')
    df_mini = pd.read_csv('hpo-data/mini/mini/args_table.csv')
    df_extended = pd.read_csv('hpo-data/extended/extended/unnormalized_args_table.csv')

    # Load meta features
    with open(META_PATH) as f:
        meta = json.load(f)

    meta_lookup = {}
    for name, desc in zip(meta['dataset_names'], meta['dataset_descriptors']):
        meta_lookup[name] = desc[:2]
    
    # Swap one-hot columns for meta features
    def process_df(df):
        dataset_cols = [col for col in df.columns if 'cat__dataset' in col]

        def get_meta_features(row):
            for col in dataset_cols:
                if row[col] == 1.0:
                    name = col.replace('cat__dataset_mtlbm/', '')
                    return meta_lookup.get(name, [0,0])
            return [0,0]
        
        meta_features = df.apply(get_meta_features, axis=1, result_type='expand')
        meta_features.columns = ['num_images', 'num_classes']

        df = df.drop(columns=dataset_cols)
        df['num_images'] = meta_features['num_images']
        df['num_classes'] = meta_features['num_classes']
        return df

    df_micro = process_df(df_micro)
    df_mini = process_df(df_mini)
    df_extended = process_df(df_extended)

    # Combine all three
    df = pd.concat([df_micro, df_mini, df_extended], ignore_index=True)
    df = df.dropna(axis=1)

    # Drop useless columns
    to_drop = ['Unnamed: 0', 'max_eval_top5', 'distributed', 'prefetcher', 'rank', 'world_size']
    existing_drops = [col for col in to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)

    X = df.drop(columns=['max_eval_top1']) # Everything the model gets to see (hyperparameters etc)
    y = df['max_eval_top1'] # Accuracy we want it to predict

    # Split into training and 20% test with reproducible results (seed = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Normalize training data
    X_test = scaler.transform(X_test) # Normalize testing data from training normalization

    return X_train, X_test, y_train, y_test

def load_hpob_data(search_space='6767'):
    # Load all HPO-B data for this search space
    with open('hpo-data/hpob-data/meta-train-dataset.json') as f:
        train_data = json.load(f)
    with open('hpo-data/hpob-data/meta-test-dataset.json') as f:
        test_data = json.load(f)
    with open('hpo-data/hpob-data/meta-validation-dataset.json') as f:
        val_data = json.load(f)

    # Combine all splits to get maximum data, then do our own 80/20 split
    all_sources = [train_data, test_data, val_data]

    # Collect all dataset IDs across all splits for one-hot encoding
    all_ds_ids = set()
    for source in all_sources:
        if search_space in source:
            all_ds_ids.update(source[search_space].keys())
    ds_ids = sorted(all_ds_ids)
    ds_to_idx = {ds: i for i, ds in enumerate(ds_ids)}

    # Extract rows with dataset ID one-hot features
    X_all, y_all = [], []
    for source in all_sources:
        if search_space in source:
            for ds_id in source[search_space]:
                ds_onehot = [0] * len(ds_ids)
                ds_onehot[ds_to_idx[ds_id]] = 1
                for i in range(len(source[search_space][ds_id]['y'])):
                    x = source[search_space][ds_id]['X'][i] + ds_onehot
                    X_all.append(x)
                    y_all.append(source[search_space][ds_id]['y'][i][0])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

class HPODataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)

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

def get_all_datasets():
    
    X_train, X_test, y_train, y_test = load_all_data()

    train_dataset = HPODataset(X_train, y_train)
    test_dataset = HPODataset(X_test, y_test)

    return train_dataset, test_dataset

def get_hpob_datasets():
    X_train, X_test, y_train, y_test = load_hpob_data()
    train_dataset = HPODataset(X_train, y_train)
    test_dataset = HPODataset(X_test, y_test)
    return train_dataset, test_dataset

def load_unified_hpob_data():
    # Load all HPO-B data across all search spaces
    all_sources = []
    for name in ['meta-train-dataset.json', 'meta-test-dataset.json', 'meta-validation-dataset.json']:
        with open(f'hpo-data/hpob-data/{name}') as f:
            all_sources.append(json.load(f))

    # Get all search spaces and their feature counts
    search_spaces = sorted(all_sources[0].keys())
    ss_to_idx = {ss: i for i, ss in enumerate(search_spaces)}
    ss_feat_counts = {}
    for ss_id in search_spaces:
        ds_id = list(all_sources[0][ss_id].keys())[0]
        ss_feat_counts[ss_id] = len(all_sources[0][ss_id][ds_id]['X'][0])

    # Collect all dataset IDs across all search spaces and splits
    all_ds_ids = set()
    for source in all_sources:
        for ss_id in source:
            all_ds_ids.update(source[ss_id].keys())
    ds_ids = sorted(all_ds_ids)
    ds_to_idx = {ds: i for i, ds in enumerate(ds_ids)}
    n_datasets = len(ds_ids)

    # Pad all X to max feature size (18) + add dataset one-hot + search space index
    max_features = 18
    X_all, y_all, ss_idx_all = [], [], []

    for source in all_sources:
        for ss_id in source:
            ss_idx = ss_to_idx[ss_id]
            n_feat = ss_feat_counts[ss_id]
            for ds_id in source[ss_id]:
                ds_onehot = [0] * n_datasets
                ds_onehot[ds_to_idx[ds_id]] = 1
                for i in range(len(source[ss_id][ds_id]['y'])):
                    x = source[ss_id][ds_id]['X'][i]
                    x_padded = x + [0.0] * (max_features - n_feat)
                    x_full = x_padded + ds_onehot
                    X_all.append(x_full)
                    y_all.append(source[ss_id][ds_id]['y'][i][0])
                    ss_idx_all.append(ss_idx)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    ss_idx_all = np.array(ss_idx_all, dtype=np.int64)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, ss_train, ss_test = train_test_split(
        X_all, y_all, ss_idx_all, test_size=TEST_SIZE, random_state=SEED
    )

    # Scale only the first 18 columns (hyperparameters), not the dataset one-hots
    scaler = StandardScaler()
    X_train[:, :max_features] = scaler.fit_transform(X_train[:, :max_features])
    X_test[:, :max_features] = scaler.transform(X_test[:, :max_features])

    return X_train, X_test, y_train, y_test, ss_train, ss_test, ss_feat_counts, search_spaces, n_datasets

class UnifiedHPODataset(Dataset):
    def __init__(self, X, y, ss_idx):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ss_idx = torch.tensor(ss_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ss_idx[idx]

def get_unified_hpob_datasets():
    X_train, X_test, y_train, y_test, ss_train, ss_test, ss_feat_counts, search_spaces, n_datasets = load_unified_hpob_data()
    train_dataset = UnifiedHPODataset(X_train, y_train, ss_train)
    test_dataset = UnifiedHPODataset(X_test, y_test, ss_test)
    # Return info the model needs to build per-search-space input layers
    ss_info = {ss_to_idx: ss_feat_counts[ss] for ss_to_idx, ss in enumerate(search_spaces)}
    return train_dataset, test_dataset, ss_info, n_datasets