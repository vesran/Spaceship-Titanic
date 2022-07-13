"""
Script to split data.
"""
import logging

import config
from dataprep.pipeline import get_data


def run_split():
    logging.info('Splitting data...')

    # Read data
    df = get_data(mode='train')
    logging.info('Raw/Intermediate data imported.')

    # Split
    logging.info(f'Start splitting with train%={config.TRAIN_PERCENT}')
    df['InTrain'] = df['PassengerId'].apply(lambda x: hash(x) % 100 < config.TRAIN_PERCENT)
    df_train = df.query('InTrain').drop('InTrain', axis=1).reset_index(drop=True)
    df_val = df.query('~InTrain').drop('InTrain', axis=1).reset_index(drop=True)
    logging.info('Data splitted')

    # Save
    logging.info('Saving data')
    df_train.to_csv(config.PATH_TO_TRAIN_SPLIT, index=False)
    df_val.to_csv(config.PATH_TO_VAL_SPLIT, index=False)
    logging.info(f'Data saved to \n--Train-- {config.PATH_TO_TRAIN_SPLIT} \n--Val-- {config.PATH_TO_VAL_SPLIT}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        encoding='utf-8', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("logs.log"),
                            logging.StreamHandler()
                        ], 
                        level=logging.INFO)
    # Run main function
    run_split()
