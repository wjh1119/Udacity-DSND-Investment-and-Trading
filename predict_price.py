import sys
from finance.stock_data import StockData
from finance.model import StockModel, ModelData, get_rolling_data

import pandas as pd
import numpy as np


def predict_price(code,
                  train_start_date,
                  train_end_date,
                  predict_date,
                  save_file_path=None,
                  verbose=0):
    '''predict close price of the stock or fund in "predict_data"
    
    Parameters
    -----------------------------
    train_start_date,train_end_date,predict_data: date-like string
    save_file_path: str
        where the model saves
    verbose: 1 or 0
        1: show the detail of process
        0: don't show the detail of process

    Returns:
    float: the close price which the model predict  
    '''

    # Predcit on the nth day of the end of the training window
    predict_period = len(pd.bdate_range(train_end_date, predict_date)) - 1

    # training window
    train_period = predict_period * 6

    # model save path
    if not save_file_path:
        save_file_path = 'models/%s_%s_%s_%s.hdf5' % (code, train_start_date,
                                                      train_end_date,
                                                      predict_date)

    # get data of prices
    print("Getting data of prices of %s between %s and %s" %
          (code, train_start_date, train_end_date))

    sd = StockData()
    price_data = sd.get_price(
        code, start_date=train_start_date, end_date=train_end_date)[1]
    print(
        "Successfully acquired data, the data of prices has %d samples and %d features"
        % (price_data.shape[0], price_data.shape[1]))

    # process data
    price_data = price_data.apply(lambda x: np.log(x), axis=1)

    # get rolling dataset
    rolling_X, rolling_y = get_rolling_data(
        price_data, train_period=train_period, predict_period=predict_period)

    # modify and split data for model's training
    model_data = ModelData(rolling_X, rolling_y, seed=666, shuffle=False)
    model_train_data, model_validate_data, model_test_data = model_data.train_validate_test_split(
        test_size=0.20, validate_size=0.20)

    # trian model
    print("Training model, please be patient.......")
    sm = StockModel(timesteps=train_period, input_dim=5)
    LSTM_model = sm.train(
        model_train_data,
        model_validate_data,
        save_file_path=save_file_path,
        verbose=verbose)

    X_predict = np.array([price_data[-train_period:].values.tolist()])

    y_predict = LSTM_model.predict(X_predict)

    # show the predict result
    print(
        "The model predicts that the closing price of %s on %s will be %.2f yuan."
        % (code, predict_date, np.exp(y_predict[0][0])))
    return np.exp(y_predict[0][0])


if __name__ == '__main__':
    if len(sys.argv) == 5:
        code = sys.argv[1]
        train_start_date = sys.argv[2]
        train_end_date = sys.argv[3]
        predict_date = sys.argv[4]

        predict_price(code, train_start_date, train_end_date, predict_date)
