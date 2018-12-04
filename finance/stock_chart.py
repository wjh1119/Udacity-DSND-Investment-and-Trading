from pyecharts import Kline
import pandas as pd


def kline_chart(df_or_dfs, chart_name):

    kline = Kline(chart_name, width="100%")

    def change_df_to_chart_data(df):

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
            "日K",
            date,
            data,
            mark_point=["max"],
            is_datazoom_show=True,
        )
    elif type(df_or_dfs) == list:

        for df_key in df_or_dfs.keys():
            
            date, data = change_df_to_chart_data(df_or_dfs[df_key])

            kline.add(
                df_key,
                date,
                data,
                mark_point=["max"],
                is_datazoom_show=True,
            )

    return kline