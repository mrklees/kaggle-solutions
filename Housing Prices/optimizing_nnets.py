import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import keras as K
from keras.layers import Input 
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model 

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, randint, conditional

def preprocess_data():
    def encode_one_categorical_feature(column):
        le = LabelEncoder()
        ohe = OneHotEncoder(sparse=False)
        num_encoded = le.fit_transform(column.fillna('unk'))
        oh_encoded = ohe.fit_transform(num_encoded.reshape(-1, 1))
        return oh_encoded
        
    data = pd.read_csv('data/train.csv')
    target = ['SalePrice']
    features = data.drop(['Id'] + target, axis=1).columns
    
    dataset_types = pd.DataFrame(data[features].dtypes, columns=['datatype'])
    dataset_types.reset_index(inplace=True)

    numeric_features = dataset_types.rename(columns={"index" : "feature"}).feature[(dataset_types.datatype == 'float64') | (dataset_types.datatype == 'int64')]
    num_data = data[numeric_features]
    num_features = num_data.fillna(num_data.mean()).values
    scaler = StandardScaler()
    num_features_scaled = scaler.fit_transform(num_features)

    categorical_features = dataset_types.rename(columns={"index" : "feature"}).feature[(dataset_types.datatype == 'object')]
    cat_data = data[categorical_features]
    cat_features = np.hstack([encode_one_categorical_feature(data[column]) for column in cat_data.columns])
    
    X = np.hstack((num_features_scaled, cat_features))
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=606)
    return X_train, X_test, y_train, y_test
    
def keras_model(X_train, X_test, y_train, y_test):
    NUM_EPOCHS = 125
    BATCH_SIZE = 128
    
    inputs = Input(shape=(304, ))
    x = Dropout({{uniform(0.1, 0.5)}})(inputs)
    
    x = Dense({{choice([64, 128, 256])}})(x)
    x = Activation("relu")(x)
    x = Dropout({{uniform(0.1, 0.5)}})(x)
    
    x = Dense({{choice([64, 128, 256])}})(x)
    x = Activation("relu")(x)
    x = Dropout({{uniform(0.1, 0.5)}})(x)
    
    x = Dense({{choice([64, 128, 256])}})(x)
    x = Activation("relu")(x)
    x = Dropout({{uniform(0.1, 0.5)}})(x)
        
    predictions = Dense(1)(x)

    model = Model(inputs=[inputs], outputs=[predictions])

    model.compile(loss="mse", optimizer={{choice(["adam", "RMSprop"])}})

    score = model.evaluate(X_test, y_test, verbose=0)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=keras_model,
                                            data=preprocess_data,
                                            algo=tpe.suggest,
                                            max_evals=50,
                                            trials=Trials())
                                           
    X_train, X_test, y_train, y_test = preprocess_data()
    score = best_model.evaluate(X_test, y_test)
    print("\n The score on the test set is {:.2e}".format(score))
    print(best_run)