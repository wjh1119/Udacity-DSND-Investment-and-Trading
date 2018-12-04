import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify

from pyecharts import Kline
from finance.stock_data import StockData

app = Flask(__name__)

# load data
sd = StockData()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    price_000300_XSHG = sd.get_price("000300.XSHG")

    chart_000300_XSHG = kline_chart(price_000300_XSHG, "深证300指数")

    # render web page with plotly graphs
    return render_template(
        'master.html', chart_000300_XSHG=chart_000300_XSHG.render_embed())


# web page that handles user query and displays stock price indicator
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict price for query
    price_query = sd.get_price(query)

    if type(price_query) == int:
        return render_template(
        'go.html', query=query, error_message="Could not find any information of this code, please check it or try it later!",chart_query="")

    chart_query = kline_chart(price_query, sd.normalize_code(query))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html', query=query, error_message="", chart_query=chart_query.render_embed())


def kline_chart(df, chart_name):

    print("type of df: %s" % str(type(df)))
    date = df.index.tolist()
    data = []
    for idx in df.index:
        row = [
            df.loc[idx]['open'], df.loc[idx]['close'], df.loc[idx]['low'],
            df.loc[idx]['high']
        ]
        data.append(row)
    kline = Kline(chart_name, width="100%")
    kline.add(
        "日K",
        date,
        data,
        mark_point=["max"],
        is_datazoom_show=True,
    )

    return kline


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()