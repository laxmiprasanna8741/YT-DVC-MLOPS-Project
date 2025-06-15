import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_log_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(file_log_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path : str) -> pd.DataFrame:
    """
    Load data from a csv file 
    """

    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape of %s',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except FileNotFoundError as e:
        logger.error("file not found: %s",e)
        raise
    except Exception as e:
        logger.error('Unexpected error ocurred while loading the data: %s' ,e)
        raise

def train_model(X_train: np.ndarray,y_train: np.ndarray,params:dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    """
    try :
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('the number of samples in x_train and y_train must be the same. ')
        
        logger.debug('Initializing Randomforest model with parameters: %s',params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug('Model training completed')

        return clf
    
    except ValueError as e:
        logger.error('Valueerror during model training: %s',e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s',e)
        raise

def save_model(model, file_path:str) -> None:
    """ Save the trained model to a file. """

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s',file_path)

    except FileNotFoundError as e:
        logger.error('File path not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error ocurred while saving the model: %s',e)
        raise

def main():
    try:
        params = {'n_estimators':20,'random_state':2}
        print("main")
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        print("data")
        clf = train_model(X_train,y_train,params)
        print('model')
        model_save_path = 'model/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s',e)
        print(f'error:{e}')

if __name__ == '__main__':
    main()