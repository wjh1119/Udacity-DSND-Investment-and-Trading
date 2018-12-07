from tensorflow.python.framework import random_seed
import numpy as np

class StockModel():

    def __init__(self):
        '''initiate class
        '''
        pass

    def train(self, X, y):
        '''train model
        '''
        pass

    def load(self, load_path=None):
        '''load model
        '''
        pass

    def save(self, save_path=None):
        '''save model
        '''
        pass

    def evaluate(self, X_test, y_test):
        '''evaluate model
        '''
        pass

    def predict(self, X):
        '''predict
        '''
        pass


def get_rolling_data(X,y,train_period,predict_period):

    assert X.shape[0] == y.shape[0], (
            'X.shape: %s y.shape: %s' % (X.shape, y.shape))
    
    rolling_X, rolling_y = [],[]
    
    for i in range(len(X)-train_period-predict_period):

        curr_X=X.iloc[i:i+train_period,:]
        curr_y=y.iloc[i+train_period:i+train_period+predict_period]
        rolling_X.append(curr_X.values.tolist())
        rolling_y.append(curr_y.values.tolist())
        
    rolling_X = np.array(rolling_X)
    rolling_y = np.array(rolling_y)
    return rolling_X, rolling_y

class ModelData():
    
    def __init__(self,X,y,seed=None,shuffle=True):
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        self._seed = seed1 if seed is None else seed2
        np.random.seed(self._seed)
        
        assert X.shape[0] == y.shape[0], (
            'X.shape: %s y.shape: %s' % (X.shape, y.shape))
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
        
    def normalize(self):
        # normalize X
        self._X=(self._X-np.mean(self._X,axis=0))/np.std(self._X,axis=0)  
        
    
    def train_test_split(self,test_size=0.25):
        test_start = int(self._num_examples*(1-test_size)) + 1
        if test_start >= self._num_examples:
            pass
        train_X,test_X,train_y,test_y = self._X[:test_start],self._X[test_start:],self._y[:test_start],self._y[test_start:]

        return ModelData(train_X,train_y,self._seed), ModelData(test_X,test_y,self._seed)
        
        
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
        return self._X
    
    @property
    def y(self):
        return self._y