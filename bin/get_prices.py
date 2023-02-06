import pandas_datareader.data as pdr
import yfinance as yfin


def get_stock_data(ticker, start_date, end_date):
    """
    Gets historical stock data of given tickers between dates
    :param ticker: company, or companies whose data is to fetched
    :type ticker: string or list of strings
    :param start_date: starting date for stock prices
    :type start_date: string of date "YYYY-mm-dd"
    :param end_date: end date for stock prices
    :type end_date: string of date "YYYY-mm-dd"
    :return: stock_data.csv
    """
    i = 1
    
    yfin.pdr_override()

    all_data = pdr.DataReader(ticker, start=start_date, end=end_date)

    all_data.to_csv("~/dev/market_predict/dados/stock_prices_full.csv")

    all_data.to_csv("~/dev/market_predict/dados/" + ticker + "_dados.csv")


if __name__ == "__main__":
    yfin.pdr_override()
    df = pdr.DataReader('PETR4.SA', start='2015-01-01', end='2020-06-25')
    print(df)
