import numpy as np
import pandas as pd
from abc import ABC
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.text import Tokenizer

class Preprocessor(ABC):
    ''' Abstract preprocessor class 
    Args:
        features (list): list of feature names to train the model on
    '''
    def __init__(self, features: list):
        self.features = features
        self.scaler = StandardScaler()

    @abstractmethod
    def __call__(self):
        pass


class DescriptivesPreprocessor(Preprocessor):
'''  Preprocessor class for text descriptives '''
    def _get_X(self):
        X = df[self.features].values
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        X = self.scaler.fit_transform(X)
        return X 

    def __call__(self, 
                 train_data: pd.DataFrame, 
                 val_data: pd.DataFrame, 
                 test_data: pd.DataFrame) -> tuple[np.array, 
                                                   np.array, 
                                                   np.array]:
        """
        Args:
            train_data (pd.DataFrame): training data
            val_data (pd.DataFrame): validation data
            test_data (pd.DataFrame): test data

        Returns:
            tuple[np.array, np.array, np.array]: feature arrays
        """
        outs = []
        for d in train_data, val_data, test_data:
            outs.append(_get_X(d, self.features, self.scaler))
        return outs


class EmbeddingPreprocessor(Preprocessor):
'''  Preprocessor class for embeddings '''
    def __call__(self, 
                 train_data: pd.DataFrame, 
                 val_data: pd.DataFrame, 
                 test_data: pd.DataFrame) -> tuple[np.array, 
                                                   np.array, 
                                                   np.array]:
        """
        Args:
            train_data (pd.DataFrame): training data
            val_data (pd.DataFrame): validation data
            test_data (pd.DataFrame): test data

        Returns:
            tuple[np.array, np.array, np.array]: feature arrays
        """
        outs = []
        for d in [train_data, val_data, test_data]:
            outs.append(np.stack(d[self.features].values))
        return outs
        

class BowPreprocessor(Preprocessor):
    """  Preprocessor class for BoW models '''
        Args:
            features (list or None): column names in input dataframe
                where BoW features are encoded. If None (default), 
                builds a vocabulary and extracts BoW embeddings on the fly.
            mode (str): tfidf, binary, count or freq, see
                tf.keras.preprocessing.text.Tokenizer documentation
            n_words (int or None): number of words for Tokenizer vocabulary
            train_data (pd.DataFrame): train data
            val_data (pd.DataFrame): validation data
            test_data (pd.DataFrame): test data
            tokenizer_kwargs: kwargs for tf.keras.preprocessing.text.Tokenizer
    """
    def __init__(self, 
                 features=None, 
                 mode='tfidf', 
                 n_words=None,
                 **tokenizer_kwargs):
        self.features = features
        self.mode = mode
        self.n_words = n_words
        self.tokenizer_kwargs = tokenizer_kwargs
        if self.features is None:
            self.tokenizer = Tokenizer(num_words=n_words,
                                       **tokenizer_kwargs)
        else:
            self.tokenizer = None

    def __call__(self, 
                 train_data: pd.DataFrame, 
                 val_data: pd.DataFrame, 
                 test_data: pd.DataFrame,
                 text_col_name='text', 
                 scale=False) -> tuple[np.array, 
                                       np.array, 
                                       np.array]:
        """
        Args:
            train_data (pd.DataFrame): training data
            val_data (pd.DataFrame): validation data
            test_data (pd.DataFrame): test data
            text_col_name (str): column name for input text
            scale (bool): whether the data should be scaled

        Returns:
            tuple[np.array, np.array, np.array]: feature arrays
        """
        outs = []
        if self.features is None:
            self.tokenizer.fit_on_texts(train_texts)
            for d in train_data, val_data, test_data:
                t_texts = d[text_col_name].tolist()
                outs.append(self.tokenizer.texts_to_matrix(t, mode=self.mode))
        else:
            for d in train_data, val_data, test_data:
                vals = d[self.features].values
                if scale is True:
                     vals = self.scaler.fit_transform(vals)
                outs.append(vals)
        return outs
