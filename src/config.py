import os

ROOT = '/app'
DATA_DIR = os.path.join(ROOT, 'data')

PATH_TO_INPUT_TRAIN = os.path.join(DATA_DIR, 'input', 'train.csv')
PATH_TO_INPUT_TEST = os.path.join(DATA_DIR, 'input', 'test.csv')
PATH_TO_SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'input', 'sample_submission.csv')

PATH_TO_TRAIN_SPLIT = os.path.join(DATA_DIR, 'split', 'train.csv')
PATH_TO_VAL_SPLIT = os.path.join(DATA_DIR, 'split', 'val.csv')

PATH_TO_OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

RANDOM_SEED = 86

# Split - Percentage representing the ratio of training data, the remaninin will go to the validation dataset
TRAIN_PERCENT = 85

# Ensemble parameters
ENSEMBLE_NUM_MODELS = 51
ENSEMBLE_SAMPLE_RATIO = 0.90

# Submission parameters
SUBMISSION_VERSION = 9
