# BUILD A STOCK PRICE INDICATOR

--------------------------------------
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## 1. Installation <a name="installation"></a>  
- tensorflow(1.12.0)
- scikit-learn(0.19.1)
- keras(2.2.4)
- pyecharts(0.5.5)
- The code should run with no issues using Python versions 3.*.

## 2. Project Motivation <a name="motivation"></a>  

This project trains an LSTM model that can establish stock price indicators based on stocks and funds historical trends and integrates the entire process into a script. The LSTM model can assist stock market traders to help them make decisions about stocks and funds trading.

## 3. File Descriptions <a name="files"></a>   

> * **BUILD A STOCK PRICE INDICATOR.pdf:** the final blog post
> * **Build an LSTM model.ipynb:**  an ipython notebook that documents the modeling process
> * **predict_price.py:** a script that user can get the predicted close price by it
> * **finance/stock_data.py:** get the original data from JQData
> * **finance/stock_chart.py:** generate the Kline charts 
> * **finance/model.py:** change the data of prices into the input and output data, build an LSTM model 
> * **finance/config.ini:** a file with user name and password for user authentication
> * **models/:** a folder of the model that has been trained

## 4. Results <a name="results"></a>  
- The web app shows visualizations about data.
- The web app can use the trained model to input text and return classification results  
#### **How to run the script?**
Before run the script:
> 1. You need to apply for data trial qualification on [this website](https://www.joinquant.com/default/index/sdk?f=home&m=banner#jq-sdk-apply).
> - After opening the website, you will see the following form.
> ![Image text](https://github.com/wjh2016/Udacity-DSND-Investment-and-Trading/blob/master/readme-img/apply.png)
> - Submit the information.If there are no surprises, the API usage will be sent to your mailbox.
>
>2. After the application is successful, you can fill in the username and password into **'finance/config.ini'** in the following format. If there is no such file, please create a new one.

>         `[user]
>account = your_user_name
>password = your_password


Run the following command in the project's directory to run the script.   
>     `python predict_price.py "stock_code" "start_date_of_training_data" "the_end_date_of_training_data" "the_expected_predicted_date"`
For example:   

>     `python predict_price.py 000001.XSHG 2005-01-01 2018-11-30 2018-12-7`  

Then, the script will return the predicted value as follows:
> ![Image text](https://github.com/wjh2016/Udacity-DSND-Investment-and-Trading/blob/master/readme-img/script.png)