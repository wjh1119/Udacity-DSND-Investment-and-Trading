import numpy as np
import pandas as pd
import os
from datetime import datetime 

import configparser
from jqdatasdk import *

class StockData():
    
    def __init__(self):
        self.load_user_config()
    
    def load_user_config(self):
        '''Add user information for getting stocks' data
        '''

        cf = configparser.ConfigParser()
        cf.read('finance/config.ini')
        account = cf.get('user', 'account')
        password = cf.get('user', 'password')
        auth(account,password)
        
    
    def get_price(self, code, start_date='2005-01-01', end_date='2018-11-30'):
        '''Download stock data (adjusted close)
        
        Parameters
        ----------------------
        code: str or int
            code of stock
        start_date: str
            start_date
        end_date: str
            end_date
            
        Returns
        ----------------------
        pd.DataFrame
        '''
        try:
            code = self.normalize_code(str(code))
        except Exception as e:
            print(e)
            return -1
        
        if code == "000001.XSHE":
            price_df = pd.read_csv("finance/data/000001.SXHE.csv",index_col=0)
            price_df = price_df.loc[start_date:end_date,:]
        else:
            security_info = get_security_info(code)
            security_start_date = security_info.start_date
            security_end_date = security_info.end_date

            # compare security date and requery date
            if datetime.strptime(start_date,"%Y-%m-%d").date() < security_start_date:
                start_date = security_start_date
            if datetime.strptime(end_date,"%Y-%m-%d").date() > security_end_date:
                end_date = security_end_date

            price_df = get_price(code, start_date=start_date, end_date=end_date, frequency='daily', fields=None, 
                                      skip_paused=False, fq='pre')
        self.fill_missing_values(price_df)
        
        return price_df
    
    
    def normalize_code(self,code):
        '''Normalize code'''
        return normalize_code(code)
        

    def fill_missing_values(self,df_data):
        """Fill missing values in data frame, in place."""
        df_data.fillna(method="ffill", inplace=True)
        df_data.fillna(method="bfill", inplace=True)