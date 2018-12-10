import numpy as np
import pandas as pd
import os
import datetime

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
        normalized_code, pd.DataFrame
        '''

        try:
            normalized_code = self.normalize_code(str(code))
        except Exception as e:
            print(e)
            return [], None
        
        if normalized_code == "000001.XSHE":
            price_df = pd.read_csv("finance/data/000001.SXHE.csv",index_col=0)
            price_df = price_df.loc[start_date:end_date,:]
        else:
            security_info = get_security_info(normalized_code)
            if security_info is None:
                return [], None
            security_start_date = security_info.start_date
            security_end_date = security_info.end_date

            # compare date information of security and requery date
            if datetime.datetime.strptime(start_date,"%Y-%m-%d").date() < security_start_date:
                start_date = security_start_date
            if datetime.datetime.strptime(end_date,"%Y-%m-%d").date() > security_end_date:
                end_date = security_end_date

            price_df = get_price(normalized_code, start_date=start_date, end_date=end_date, frequency='daily', fields=None, 
                                      skip_paused=False, fq='pre')
        self.fill_missing_values(price_df)
        
        return normalized_code, price_df

    def query_prices(self,query_str, start_date='2005-01-01', end_date='2018-11-30'):

        # split query str
        query_codes = query_str.split(",")

        if len(query_codes) == 1:

            normalized_code, price_df = self.get_price(query_codes[0],start_date,end_date)
            if normalized_code:
                return [normalized_code], price_df, []
            else:
                print('Could not find price data of this code: %s' %query_codes[0])
                return [], None, [query_codes[0]]

        elif len(query_codes) >= 2:

            normalized_codes = []
            price_dfs = {}
            err_codes = []
            
            for code in query_codes:

                normalized_code, price_df = self.get_price(code,start_date,end_date)

                if normalized_code:
                    price_dfs[normalized_code] = price_df
                    normalized_codes.append(normalized_code)
                else:
                    print('Could not find price data of this code: %s' %code)
                    err_codes.append(code)
            
            if len(normalized_codes) == 0:
                return [], None, err_codes
            else:
                return normalized_codes, price_dfs,err_codes
        else:
            pass          


    def normalize_code(self,code):
        '''Normalize code'''
        return normalize_code(code)
        

    def fill_missing_values(self,df_data):
        """Fill missing values in data frame, in place."""
        df_data.fillna(method="ffill", inplace=True)
        df_data.fillna(method="bfill", inplace=True)