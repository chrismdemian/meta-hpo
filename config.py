# Values for dataset.py
DATA_PATH = 'hpo-data/micro/micro/args_table.csv'
META_PATH = 'hpo-data/dataset-meta-features/dataset-meta-features/meta-album/meta-features.json'
TEST_SIZE = 0.2
SEED = 42

# Values for model.py
INPUT_SIZE = 34
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1 # Just predicting accuracy
DROPOUT = 0.2 # To prevent overfitting

# Values for train.py
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
EPOCHS = 50
LR_STEP_SIZE = 30 # Reduce learning rate every 100 epochs
LR_GAMMA = 0.5 # Half the learning rate
WEIGHT_DECAY = 1e-4