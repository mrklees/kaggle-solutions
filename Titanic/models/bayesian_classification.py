# -*- coding: utf-8 -*-#
from random import randint
# Data Structures
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler

class TitanicData(object):
    """Titanic Data
    
    This class will contain the entire data pipeline from raw data to prepared 
    numpy arrays.  It's eventually inherited by the model class, but is left 
    distinct for visibility.  Filepath is hardcoded in, but is easy to change given  
    our context. Sorry, I'm lazy :D
    """
    filepath = "C:/Users/perus/OneDrive/Documents/GitHub/kaggle-solutions/Titanic/data/"
    train_fn = 'train.csv'
    test_fn = 'test.csv'
    
    def __init__(self):
        self.X_train, self.y_train, self.X_valid, self.y_valid = self.preproc()
    
    def import_and_split_data(self):
        X, y = self.select_features(pd.read_csv(self.filepath + self.train_fn))
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 606, stratify = y)
        return X_train, y_train, X_valid, y_valid
        
    def select_features(self, data):
        """Selects the features that we'll use in the model. Drops unused features"""
        target = ['Survived']
        features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        dropped_features = ['Cabin', 'Ticket']
        X = data[features].drop(dropped_features, axis=1)
        y = data[target]
        return X, y
    
    def fix_na(self, data):
        """Fill na's with the mean (in the case of fare), and with C in the case of embarked"""
        na_vars = {"Fare" : data.Fare.mean(), "Embarked" : "C", "Age" : data.Age.mean()}
        return data.fillna(na_vars)
    
    def create_dummies(self, data, cat_vars, cat_types):
        """Processes categorical data into dummy vars"""
        cat_data = data[cat_vars].values
        for i in range(len(cat_vars)):   
            bins = LabelBinarizer().fit_transform(cat_data[:, 0].astype(cat_types[i]))
            cat_data = np.delete(cat_data, 0, axis=1)
            cat_data = np.column_stack((cat_data, bins))
        return cat_data
    
    
    def standardize(self, data, real_vars):
        """Processes numeric data"""
        real_data = data[real_vars]
        scale = StandardScaler()
        return scale.fit_transform(real_data)
    
    def extract_titles(self, data):
        """Extract titles from the Name field and create appropriate One Hot Encoded Columns"""
        title_array = data.Name
        first_names = title_array.str.rsplit(', ', expand=True, n=1)
        titles = first_names[1].str.rsplit('.', expand=True, n=1)
        known_titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
           'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',
           'Jonkheer']
        for title in known_titles:
            try:
                titles[title] = titles[0].str.contains(title).astype('int')
            except:
                titles[title] = 0
        return titles.drop([0,1], axis=1).values
    
    def preproc(self):
        """Executes the full preprocessing pipeline."""
        # Import Data & Split
        X_train_, y_train, X_valid_, y_valid = self.import_and_split_data()
        # Fill NAs
        X_train, X_valid = self.fix_na(X_train_), self.fix_na(X_valid_)
        
        # Preproc Categorical Vars
        cat_vars = ['Pclass', 'Sex', 'Embarked']
        cat_types = ['int', 'str', 'str']
        X_train_cat, X_valid_cat = self.create_dummies(X_train, cat_vars, cat_types), self.create_dummies(X_valid, cat_vars, cat_types)
        # Preprocess Numeric Vars
        real_vars = ['Fare', 'SibSp', 'Parch', 'Age']
        X_train_real, X_valid_real = self.standardize(X_train, real_vars), self.standardize(X_valid, real_vars)
        # Extract Titles
        X_train_titles, X_valid_titles = self.extract_titles(X_train), self.extract_titles(X_valid)
        # Recombine
        X_train, X_valid = np.column_stack((X_train_cat, X_train_real, X_train_titles)), np.column_stack((X_valid_cat, X_valid_real, X_valid_titles))
        # Fill Missing Ages
        #X_train, X_test = impute_ages(X_train), impute_ages(X_test)
        return X_train.astype('float32'), y_train.values, X_valid.astype('float32'), y_valid.values
    
class TitanicModel(TitanicData):
    
    def __init__(self):
        self.X_train, self.y_train, self.X_valid, self.y_valid = self.preproc()
        
    def build_single_model(self, random_state, num_layers, verbose=True):
        """Create a single neural network with variable layers
        
        This function will both assign the model to the self.model attribute, as well
        as return the model.  I'm pretty afraid of side effects resulting from 
        changing the state within the object, but then it hasn't ruined by day yet...
        """
        model = MLPClassifier(
                        hidden_layer_sizes=(1024, ) * num_layers,
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size=100,
                        max_iter=64,
                        learning_rate_init=0.001,
                        random_state=random_state,
                        early_stopping=True,
                        verbose=verbose
                        )
        self.model = model
        return model
        
    def fit(self):
        """Fit the model to the training data"""
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """Score the model against the validation data"""
        return self.model.score(self.X_valid, self.y_valid)
        
    def build_voting_model(self, model_size=10, n_jobs=1):
        """Build a basic Voting ensamble of neural networks with various seeds and numbers of layers
        
        The idea is that we'll generate a large number of neural networks with various depths 
        and then aggregate across their beliefs.
        """
        models = [(seed, self.build_single_model(seed, randint(1, 5))) for seed in np.random.randint(1, 1e6, size=model_size)]
        ensamble = VotingClassifier(models, voting='soft', n_jobs=n_jobs)
        self.model = ensamble
        return ensamble
    
    def build_bagging_model(self):
        """Apply Bagging to our neural network as an another way to improve performance"""
        ensamble = BaggingClassifier(base_estimator=self.build_single_model(random_state=606, num_layers=4))
        self.model = ensamble
        return ensamble
    
        