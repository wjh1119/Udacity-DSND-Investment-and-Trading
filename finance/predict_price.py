import sys
from finance.stock_data import StockData
from finance.model import StockModel, ModelData, get_rolling_data

import pandas as pd
import numpy as np


def predict_price(code,
                  train_start_date,
                  train_end_date,
                  predict_date,
                  save_file_url=None,
                  verbose=0):

    n_day_later_predict = len(pd.bdate_range(train_end_date, predict_date)) - 1
    train_period = n_day_later_predict * 6
    predict_period = 1

    # get price data
    sd = StockData()
    price_data = sd.get_price(
        code, start_date=train_start_date, end_date=train_end_date)[1]

    # process data
    price_data = price_data.apply(lambda x: np.log(x), axis=1)

    # get rolling dataset
    rolling_X, rolling_y = get_rolling_data(
        price_data,
        price_data.iloc[:, 1],
        train_period=train_period,
        predict_period=predict_period)

    model_data = ModelData(rolling_X, rolling_y, seed=666, shuffle=False)
    model_train_data, model_validate_data, model_test_data = model_data.train_validate_test_split(
        test_size=0.20, validate_size=0.20)

    sm = StockModel()
    LSTM_model = sm.train_model(
        model_train_data,
        model_validate_data,
        save_file_url=save_file_url,
        verbose=verbose)

    X_predict = np.array([price_data[-train_period:].values.tolist()])

    y_predict = LSTM_model.predict(X_predict)

    return np.exp(y_predict[0][0])


if __name__ == '__main__':
    if len(sys.argv) == 5:
        code = sys.argv[1]
        train_start_date = sys.argv[2]
        train_end_date = sys.argv[3]
        predict_price = sys.argv[4]

        print(
            predict_price(code, train_start_date, train_end_date,
                          predict_date))
