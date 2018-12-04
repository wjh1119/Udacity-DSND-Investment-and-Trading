import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify

from finance.stock_chart import kline_chart
from finance.stock_data import StockData

app = Flask(__name__)

# load data
sd = StockData()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    normalized_name, price_000300_XSHG = sd.get_price("000300.XSHG")

    chart_000300_XSHG = kline_chart(price_000300_XSHG, "深证300指数").render_embed()

    # render web page with plotly graphs
    return render_template(
        'master.html', chart_000300_XSHG=chart_000300_XSHG)


# web page that handles user query and displays stock price indicator
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict price for query
    normalized_codes, query_data, error_codes = sd.query_prices(query)

    chart_query = None
    error_message = ""

    if len(normalized_codes) == 0:
        chart_query = ""
    elif len(normalized_codes) == 1:
        chart_query = kline_chart(query_data, normalized_codes[0]).render_embed()
    elif len(normalized_codes) > 1:
        chart_query = kline_chart(query_data, ",".join(normalized_codes)).render_embed()

    if len(error_codes) > 0:
        error_message = "Could not find any information of these codes: %s, please check them!" %str(error_codes)
    
    return render_template(
        'go.html', query=query, error_message=error_message,chart_query=chart_query)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()