from pyecharts import Kline
import pandas as pd


def kline_chart(df_or_dfs, chart_name):
    '''Create a Kline chart by given data and chart's name
    
    Parameters
    ---------------------------------
    df_or_dfs: pd.DataFrame or list of pd.DataFrame
        chart's data
    chart_name: str
        chart's name

    Returns
    --------------------------------
    Kline chart or list of Kline chart 
    '''

    kline = Kline(chart_name, width="100%")

    def change_df_to_chart_data(df):
        '''Process raw data into data suitable for generating charts
        
        Parameters
        -------------------------
        df: pd.DataFrame
            raw data
        
        Returns
        -------------------------
        date: list of date
        data: daily data including opening price, closing price, low price and high price
        '''
        date = df.index.tolist()
        data = []
        for idx in df.index:
            row = [
                df.loc[idx]['open'], df.loc[idx]['close'], df.loc[idx]['low'],
                df.loc[idx]['high']
            ]
            data.append(row)
        return date, data

    if type(df_or_dfs) == pd.DataFrame:

        date, data = change_df_to_chart_data(df_or_dfs)

        kline.add(
            "daily",
            date,
            data,
            is_datazoom_show=True,
        )
    elif type(df_or_dfs) == dict:

        for df_key in df_or_dfs.keys():
            
            date, data = change_df_to_chart_data(df_or_dfs[df_key])

            kline.add(
                df_key,
                date,
                data,
                is_datazoom_show=True,
            )

    return kline