import logging

import numpy as np
from sklearn.metrics import accuracy_score

import config
from features.engineer import FeatureEngineer


class Pool:

    def __init__(self, df_train, func_model, n_models=10, frac=0.9) -> None:
        # func_model : create_model()
        self.models = self._build_models(func_model, n_models)
        self.datasets, self.vals = self._build_datasets(df_train, n_models, frac)
        self.engineers = self._build_engineers(self.datasets)


    def _build_models(self, func_model, n_models):
        logging.info(f'Building {n_models} models')
        return [func_model() for _ in range(n_models)]


    def _build_datasets(self, df_train, n_datasets, frac):
        logging.info(f'Building {n_datasets} datasets')
        trains, vals = [], []
        n_train = int(frac * len(df_train))
        for i in range(n_datasets):
            sample = df_train.sample(frac=1, random_state=config.RANDOM_SEED + i).reset_index(drop=True)
            sample_train = sample.head(n_train).reset_index(drop=True)
            sample_val = sample.tail(len(sample) - n_train).reset_index(drop=True)
            trains.append(sample_train)
            vals.append(sample_val)
        return trains, vals


    def _build_engineers(self, datasets):
        logging.info(f'Building {len(datasets)} engineers')
        return [FeatureEngineer(train_sample) for train_sample in datasets]


    def train_and_eval(self):
        logging.info('Train and eval')
        scores = []
        for train_sample, engineer, model, val_sample in zip(self.datasets, self.engineers, self.models, self.vals):
            # Get features
            X_train = engineer.process_features(train_sample, train=True)
            X_val = engineer.process_features(val_sample, train=False)

            # Get labels
            y_train = train_sample['Transported']
            y_val = val_sample['Transported']

            # Fit
            model.fit(X_train, y_train)

            # Evaluate
            preds = model.predict(X_val).flatten()
            general_score = accuracy_score(preds, y_val)
            scores.append(general_score)
            logging.info(f"Submodel accuracy = {general_score}")
            
        logging.info(f"Mean accuracy = {np.mean(scores)}")
        return scores


    def predict_and_examine(self, df_test):
        output = df_test.filter(['PassengerId', 'Transported'])
        for i, (engineer, model) in enumerate(zip(self.engineers, self.models)):
            # Get features
            X_test = engineer.process_features(df_test, train=False)

            # Predict
            preds = model.predict(X_test).flatten()
            output[f'pred_{i}'] = preds
        
        # Compute final predictions
        pred_cols = [col for col in output.columns if col.startswith('pred_')]
        output['final_pred'] = (output[pred_cols].mean(axis=1) + 0.5).apply(np.floor, axis=1).astype(bool)
        return output


    def predict(self, df_test):
        return self.predict_and_examine(df_test)['final_pred']