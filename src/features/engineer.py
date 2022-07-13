import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import prince

from dataprep.pipeline import get_split, get_data
from models.evaluate import get_confusion_matrix
import config



class FeatureEngineer:

    def __init__(self, df_train) -> None:
        self.fitted = False
        self.mlb_home_planet = MultiLabelBinarizer()
        self.scaler_age = MinMaxScaler()
        self.age_median = df_train.Age.median()
        self.common_destination = df_train.Destination.mode().item()
        self.mlb_destination = MultiLabelBinarizer()

        self.scaler_bill = MinMaxScaler()
        self.mlb_cabin = MultiLabelBinarizer()
        self.scaler_cabin = MinMaxScaler()
        self.cabin_num_median = df_train.CabinNum.median()

        self.decks = pd.crosstab(df_train['CabinDeck'], df_train['Transported'])
        self.decks = ((self.decks[1] / self.decks.sum(axis=1))
                        .reset_index()
                        .rename({0: 'transported_ratio'}, axis=1)
                        )

        self.pca = prince.PCA(n_components=3)
        self.mca = prince.MCA(n_components=4)


    
    def _filter_features_columns(self, df):
        cols = [col for col in df.columns if col.startswith('feat_')]
        cols.append('PassengerId')
        return df.filter(cols)


    def _impute_home_planet(self, df):
        df = df.copy()
        
        def impute(row):
            if type(row['HomePlanet']) == str:
                return row['HomePlanet']
            else:
                if row['CabinDeck'] in ('A', 'B', 'C', 'T'):
                    return 'Europa'
                elif (row['Destination'] == 'PSO J318.5-22') or (row['CabinDeck'] == 'G'):
                    return 'Earth'
                elif row['RoomService'] >= 900:
                    return 'Mars'
                elif row['VIP']:
                    return 'Europa'
                else:
                    return 'Earth'
        
        df['HomePlanet'] = df.apply(impute, axis=1)
        return df


    def process_home_planet(self, df, train=False):
        df = self._impute_home_planet(df)
        df = df.filter(['PassengerId', 'HomePlanet'])
        #df['HomePlanet'] = df.HomePlanet.fillna('Earth')
        seq = df.HomePlanet.values.reshape(-1, 1)
        if train:
            self.mlb_home_planet.fit(seq)
        feats = self.mlb_home_planet.transform(seq)
        for i, c in enumerate(self.mlb_home_planet.classes_):
            df[f'feat_HomePlanet={c}'] = feats[:, i]
        return self._filter_features_columns(df)


    def process_cryosleep(self, df, train=False):
        df = df.filter(('PassengerId', 'CryoSleep'))
        df['CryoSleep'] = df.CryoSleep.fillna(False)  # Mode
        df['feat_CyroSleep'] = df['CryoSleep'].astype(int)
        return self._filter_features_columns(df)


    def process_destination(self, df, train=False):
        df = df.filter(['PassengerId', 'Destination'])
        df['Destination'] = df.Destination.fillna(self.common_destination)
        if train:
            self.mlb_destination.fit(df['Destination'].values.reshape(-1, 1))
        feats = self.mlb_destination.transform(df['Destination'].values.reshape(-1, 1))
        for i, col in enumerate(self.mlb_destination.classes_):
            df[f'feat_{col}'] = feats[:, i]
        return self._filter_features_columns(df)


    def process_vip(self, df, train=False):
        df = df.filter(['PassengerId', 'VIP'])
        df['VIP'] = df.VIP.fillna(False)
        df['feat_VIP'] = df.VIP.astype(int)
        return self._filter_features_columns(df)


    def process_age(self, df, train=False):
        df = df.filter(('PassengerId', 'Age'))
        df['Age'] = df.Age.fillna(self.age_median)
        ages = df.Age.values.reshape(-1, 1)
        if train:
            self.scaler_age.fit(ages)
        df['feat_age'] = self.scaler_age.transform(ages)
        df['feat_is_child'] = (df['Age'] < 18).astype(int)
        return self._filter_features_columns(df)


    def process_bill(self, df, train=False):
        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df = df.filter(['PassengerId'] + cols)
        df = df.fillna(0.)
        
        if train:
            self.scaler_bill.fit(df[cols].values)
        
        norm_bills = self.scaler_bill.transform(df[cols].values)
        feat_cols = []
        for i, col in enumerate(cols):
            feat_col = f'feat_{col}'
            df[feat_col] = norm_bills[:, i]
            feat_cols.append(feat_col)
        df['feat_total_norm_bill'] = df[feat_cols].sum(axis=1)
        return self._filter_features_columns(df)


    def process_cabin(self, df, train=False):
        cat_cols = ['CabinSide']
        df = df.filter(['PassengerId'] + cat_cols + ['CabinNum', 'CabinDeck'])
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Handle Cabin nums & side
        df['CabinNum'] = df['CabinNum'].fillna(self.cabin_num_median)
        if train:
            self.mlb_cabin.fit(df[cat_cols].values)
            self.scaler_cabin.fit(df['CabinNum'].values.reshape(-1, 1))
        feats = self.mlb_cabin.transform(df[cat_cols].values)
        for i, c in enumerate(self.mlb_cabin.classes_):
            df[f'feat_{c}'] = feats[:, i]
        df['feat_CabinNum'] = self.scaler_cabin.transform(df['CabinNum'].values.reshape(-1, 1)).flatten()
        
        # Add Decks features
        df = df.merge(self.decks, on='CabinDeck', how='left')
        df['feat_transported_ratio_cabin_deck'] = df['transported_ratio'].fillna(0.5)
        return self._filter_features_columns(df)


    def extract_features_from_continuous(self, feats, train=False):
        feats = feats.filter(['feat_age', 'feat_RoomService', 'feat_FoodCourt', 
                            'feat_ShoppingMall', 'feat_Spa', 'feat_VRDeck', 
                            'feat_total_norm_bill', 'feat_CabinNum', 
                            'feat_transported_ratio_cabin_deck', 
                            'PassengerId'])
        if train:
            self.pca.fit(feats.drop('PassengerId', axis=1))
        dimred = self.pca.transform(feats.drop('PassengerId', axis=1))
        df = pd.DataFrame({'PassengerId': feats['PassengerId']})
        for col in dimred.columns:
            df[f'feat_pca_{col}'] = dimred[col]
        return self._filter_features_columns(df.reset_index())


    def extract_features_from_categorical(self, feats, train=False):
        feats = feats.filter(['PassengerId', 
                            'feat_HomePlanet=Earth', 'feat_HomePlanet=Europa',
                            'feat_HomePlanet=Mars', 'feat_CyroSleep', 'feat_is_child', 
                            'feat_VIP', 'feat_P', 'feat_S'
                            ])
        if train:
            self.mca.fit(feats.drop('PassengerId', axis=1))
        dimred = self.mca.transform(feats.drop('PassengerId', axis=1))
        df = pd.DataFrame({'PassengerId': feats['PassengerId']})
        for col in dimred.columns:
            df[f'feat_mca_{col}'] = dimred[col]
        return self._filter_features_columns(df.reset_index())
        

    def process_features(self, df, train=False):
        feats = df.filter(['PassengerId'])
        feats = feats.merge(self.process_home_planet(df, train=train), on='PassengerId', how='left')
        feats = feats.merge(self.process_cryosleep(df, train=train), on='PassengerId', how='left')
        feats = feats.merge(self.process_age(df, train=train), on='PassengerId', how='left')
        #feats = feats.merge(self.process_destination(df, train=train), on='PassengerId', how='left')
        feats = feats.merge(self.process_vip(df, train=train), on='PassengerId', how='left')
        feats = feats.merge(self.process_bill(df, train=train), on='PassengerId', how='left')
        feats = feats.merge(self.process_cabin(df, train=train), on='PassengerId', how='left')
        
        #feats = feats.merge(self.extract_features_from_continuous(feats, train=train), on='PassengerId', how='left')
        #feats = feats.merge(self.extract_features_from_categorical(feats, train=train), on='PassengerId', how='left')
        self.fitted = True
        return feats.set_index('PassengerId')