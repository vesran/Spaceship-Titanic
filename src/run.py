import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier

from dataprep.pipeline import get_split, get_data
from utils.submit import make_submission
from models.ensemble import Pool
import config


def create_model():
    model = CatBoostClassifier(iterations=80,
                          learning_rate=0.2,
                          depth=10, 
                          loss_function='Logloss', 
                          random_seed=config.RANDOM_SEED)
    return model


def run():
    df_train = get_data(mode='train')
    pool = Pool(df_train, create_model, n_models=config.ENSEMBLE_NUM_MODELS, frac=config.ENSEMBLE_SAMPLE_RATIO)
    scores = pool.train_and_eval()

    logging.info(f'Scores : {np.mean(scores)}')

    make_submission(pool)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        encoding='utf-8', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("logs.log"),
                            logging.StreamHandler()
                        ], 
                        level=logging.INFO)
    run()