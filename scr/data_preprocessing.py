import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_perprocesing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_log_path = os.path.join(log_dir,'data_perprocessing.log')
file_handler = logging.FileHandler(file_log_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """ Transfroms the input text by converting it to lowercase,tokenizing, 
        removing stopwords and punctuation, and stemming. """
    
    try:
        ps = PorterStemmer()
        # convert to lowercase
        text = text.lower()
        # Tokenize the text 
        text = nltk.word_tokenize(text)
        # remove non - alphanumeric tokens
        text = [word for word in text if word.isalnum()]
        # remove stopwords and punctuation
        text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        #steam the words
        text = [ps.stem(word) for word in text]
        # join the tokens back into a single string
        return " ".join(text)
    except Exception as e:
        logger.debug('error during text transformation')
        raise

def preprocess_df(df,text_columns='text',target_column='target'):
    """
    Preprocess the dataframe by endecoding the target column, 
    removing duplicates,and transforming the text columns. """
    try :
        logger.debug('starting preprocessing for Dataframe')

        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed')

        df.loc[:,text_columns] = df[text_columns].apply(transform_text)
        logger.debug('text column transformed')

        return df
    
    except KeyError as e:
        logger.error('column not found %s',e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s',e)
        raise

def main(text_column='text',target_column='target'):
    """ Main function to load raw data, preprocess it, and save the processed data. """
    
    try:
        #fetch data from ./data/raw
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logger.debug('data loaded properly')

        #Transform the data
        train_processed_data = preprocess_df(train_data,text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)

        # store the data inside data/processed
        data_path = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug('Processed data saved to the %s',data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s',e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()


