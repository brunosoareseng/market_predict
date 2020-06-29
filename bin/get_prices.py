import pandas_datareader.data as pdr


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
    all_data = pdr.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)

    all_data.to_csv("./dados/stock_prices_full.csv")

    all_data.to_csv("./dados/" + ticker + "_dados.csv")


if __name__ == "__main__":
    get_stock_data("PETR4.SA", "2015-01-01", "2020-06-25")
