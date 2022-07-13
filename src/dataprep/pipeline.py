import pandas as pd

import config


def get_data(mode='train'):
    """
    Reads data from a raw CSV and process them.
    """
    if mode == 'train':
        path = config.PATH_TO_INPUT_TRAIN
    elif mode == 'test':
        path = config.PATH_TO_INPUT_TEST
    else:
        raise NotImplemented
        
    df = (pd.read_csv(path)
             .drop('Name', axis=1)
             .drop_duplicates(subset=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 
                                      'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 
                                      'VRDeck'] + ['Transported' if mode == 'train' else 'VRDeck'],
                              keep='first')
             .reset_index(drop=True)
         )
    # Split Cabin
    df['CabinDeck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if type(x) == str else None)
    df['CabinNum'] = df['Cabin'].apply(lambda x: x.split('/')[1] if type(x) == str else None)
    df['CabinSide'] = df['Cabin'].apply(lambda x: x.split('/')[2] if type(x) == str else None)
    
    # Process labels
    if mode != 'test':
        df['Transported'] = df['Transported'].apply(lambda x: 1 if x == True else 0)

    # Drop unused columns
    df = df.drop(['Cabin'], axis=1)
    return df


def get_split():
    """
    Reads train or validation data.
    """
    df_train = pd.read_csv(config.PATH_TO_TRAIN_SPLIT)
    df_val = pd.read_csv(config.PATH_TO_VAL_SPLIT)
    return df_train, df_val