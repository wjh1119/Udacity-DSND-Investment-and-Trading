# from finance.stock_data import StockData
from jqdatasdk import get_price, normalize_code, get_security_info, auth
import datetime
import pandas as pd
import numpy as np

# Parameter tuning
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Keras
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, Flatten, Lambda, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam, Nadam, SGD, Adadelta
from keras import backend as K

from tensorflow.python.framework import random_seed


class StockModel():
    def __init__(self, timesteps, input_dim):
        '''initiate class and build an LSTM model

        Parameters
        ----------------------------
        timesteps: Training window of the LSTM model
        input_dim: Number of features in the training data set
        '''

        # loss function
        def error_rate(y_true, y_pred):
            '''Calculate the error rate between true value and predict value'''
            return self.evaluate(y_true, y_pred)

        # build LSTM model
        def build_model(units=256, dropout=0.0, optimizer=Adam, lr=0.001):
            model = Sequential()
            model.add(LSTM(units, input_shape=(timesteps, input_dim)))
            model.add(Dropout(dropout))
            model.add(Dense(1))
            model.compile(loss=error_rate, optimizer=optimizer(lr))
            model.summary()
            return model

        # build an LSTM model
        model = build_model(units=256, dropout=0, optimizer=Adam, lr=0.001)
        self.model = model

    def train(self,
              model_train_data,
              model_validate_data,
              verbose=1,
              save_file_path=None):
        '''train model

        model_train_data: ModelData
            include X_train and y_train
        model_validate_data: ModelData
            include X_validate and y_validate
        verbose: 1 or 0
            1: show the detail of process
            0: don't show the detail of process
        '''

        y_train = model_train_data.y
        y_validate = model_validate_data.y

        X_train = model_train_data.X
        X_validate = model_validate_data.X

        # definite EarlyStopping for avoiding model overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=50, verbose=verbose, mode="auto")

        callbacks = []
        if save_file_path:
            # definite ModelCheckpoint for avoiding model overfitting
            model_checkpoint = ModelCheckpoint(
                save_file_path,
                monitor='val_loss',
                verbose=verbose,
                save_best_only=True,
                save_weights_only=False,
                mode='auto')
            callbacks = [early_stopping, model_checkpoint]

        else:
            callbacks = [early_stopping]

        self.model.fit(
            X_train,
            y_train,
            epochs=1000,
            batch_size=128,
            validation_data=(X_validate, y_validate),
            verbose=verbose,
            callbacks=callbacks)

        if save_file_path:
            self.model.load_weights(save_file_path)

        return self.model

    def evaluate(self, y_true, y_pred):
        '''Calculate the error rate between true value and predict value
        '''
        return K.mean(K.abs(K.exp(y_pred - y_true) - 1), axis=1)

    def predict(self, model_test_data):
        '''predict close price by given testing data set
        '''
        y_predict = self.model.fit(model_test_data.X)
        close_price = np.exp(y_predict)
        return close_price

    def save(self, filepath):
        '''Save model'''
        self.model.save(filepath)

    def load(self, filepath):
        '''Load model'''
        self.model = load_model(filepath)


def get_rolling_data(df, train_period, predict_period):
    '''Generating input and output data
    
    Parameters
    ------------------------------
    df: pd.DataFrame
        source data
    train_period: int
        timesteps for LSTM model
    predict_period: int
        predcit on the nth day of the end of the training window

    Returns
    ------------------------------
    input data X and output data y
    '''

    X = df
    y = df['close']

    rolling_X, rolling_y = [], []

    for i in range(len(X) - train_period - predict_period):

        curr_X = X.iloc[i:i + train_period, :]
        curr_y = y.iloc[i + train_period:i + train_period + predict_period]
        rolling_X.append(curr_X.values.tolist())
        rolling_y.append(curr_y.values.tolist())

    rolling_X = np.array(rolling_X)
    rolling_y = np.array(rolling_y)
    return rolling_X, rolling_y


class ModelData():
    '''Data for model train, predict and validate, '''

    def __init__(self, X, y, seed=None, shuffle=True):
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        self._seed = seed1 if seed is None else seed2
        np.random.seed(self._seed)

        assert X.shape[0] == y.shape[0], ('X.shape: %s y.shape: %s' %
                                          (X.shape, y.shape))
        self._num_examples = X.shape[0]

        # If shuffle
        if shuffle:
            np.random.seed(self._seed)
            randomList = np.arange(X.shape[0])
            np.random.shuffle(randomList)
            self._X, self._y = X[randomList], y[randomList]

        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def train_validate_test_split(self, validate_size=0.20, test_size=0.2):
        '''Split data into training data, validate data, test data'''

        validate_start = int(self._num_examples *
                             (1 - validate_size - test_size)) + 1
        test_start = int(self._num_examples * (1 - test_size)) + 1
        if validate_start > len(self._X) or test_start > len(self._X):
            pass
        train_X, train_y = self._X[:validate_start], self._y[:validate_start]
        validate_X, validate_y = self._X[validate_start:test_start], self._y[
            validate_start:test_start]
        test_X, test_y = self._X[test_start:], self._y[test_start:]

        if test_size == 0:
            return ModelData(train_X, train_y, self._seed), ModelData(
                validate_X, validate_y, self._seed)
        else:
            return ModelData(train_X, train_y, self._seed), ModelData(
                validate_X, validate_y, self._seed), ModelData(
                    test_X, test_y, self._seed)

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._X[start:self._num_examples]
            labels_rest_part = self._y[start:self._num_examples]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            X_new_part = self._X[start:end]
            y_new_part = self._y[start:end]
            return np.concatenate(
                (images_rest_part, X_new_part), axis=0), np.concatenate(
                    (labels_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._y[start:end]

    @property
    def X(self):
        '''Return input data set'''

        return self._X

    @property
    def y(self):
        ''' Return output data set'''

        return self._y