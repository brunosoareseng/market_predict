from datetime import date
from pandas import read_csv

from bin.get_prices import get_stock_data
from bin.price_pred_LSTM_MV import prediction


def run_sim(file, start_date, end_date):
    """
    Lê arquivos com preços e gerencia predição de preços gerando arquivos e graficos
    :param file: Arquivo com papeis para simular
    :type file: string nome do arquivo
    :return: stock_prices.csv stock_prices_full.csv
    :param start_date: starting date for stock prices
    :type start_date: string of date "YYYY-mm-dd"
    :param end_date: end date for stock prices
    :type end_date: string of date "YYYY-mm-dd"
    """

    print("\nData inicio:", start_date)
    print("Data fim:", end_date)

    dataset = read_csv(file, header=None)
    print("\nPapeis:")
    print(dataset[0].values)

    print("\nBaixando dados dos papeis...\n")

    for papel in dataset[0].values:
        get_stock_data(papel, start_date, end_date)
        print(papel + " -> OK!")

    for papel in dataset[0].values:
        print("Treinando e prevendo para: " + papel)
        prediction(papel)
        print("OK!")


if __name__ == "__main__":
    run_sim("dados/papeis.csv", "2015-01-01", date.today())
