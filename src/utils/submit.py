import os

import pandas as pd

from models.ensemble import Pool
from dataprep.pipeline import get_data
import config


def make_submission(model):
    # Test
    df_test = get_data('test')
    preds = model.predict(df_test)
    df_test['pred'] = preds

    # Make submission
    submission = (pd.read_csv(config.PATH_TO_SAMPLE_SUBMISSION)
                .drop('Transported', axis=1)
                .merge(df_test.filter(['PassengerId', 'pred']), 
                        on='PassengerId', 
                        how='left')
                .rename({'pred': 'Transported'}, axis=1)
                .astype({'Transported': bool})
                )
    output_path = os.path.join(config.PATH_TO_OUTPUT_DIR, f'sub_{str(config.SUBMISSION_VERSION).zfill(2)}.csv')
    submission.to_csv(output_path, index=False)
