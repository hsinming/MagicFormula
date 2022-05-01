#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Hsin-ming Chen
@license: GNU V3
@file: main.py
@time: 2022/05/01
@contact: hsinming.chen@gmail.com
@software: PyCharm
"""
import json
import argparse
import math
import time
import multiprocessing as mp
import concurrent.futures
from threading import Event
from collections import deque
from pathlib import Path
from typing import Generator
from ftplib import FTP
import sqlite3 as sq
import pandas as pd
import requests
from yahooquery import Ticker
from fiscalyear import FiscalDateTime


class FinancialStatement(object):
    def __init__(self, financial_dict: dict):
        self.ticker = ''
        self.financial_dict = financial_dict

    def set_ticker(self, ticker: str):
        self.ticker = ticker

    @property
    def ebit(self):
        try:
            return self.financial_dict[self.ticker]["EBIT"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def current_assets(self):
        try:
            return self.financial_dict[self.ticker]["CurrentAssets"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def current_liabilities(self):
        try:
            return self.financial_dict[self.ticker]["CurrentLiabilities"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def total_debt(self):
        try:
            return self.financial_dict[self.ticker]["TotalDebt"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def longterm_debt(self):
        try:
            return self.financial_dict[self.ticker]["LongTermDebt"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def total_cash(self):
        try:
            return self.financial_dict[self.ticker]["CashCashEquivalentsAndShortTermInvestments"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def excess_cash(self):
        try:
            return self.total_cash - max(0, self.current_liabilities - self.current_assets + self.total_cash)
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def net_working_capital(self):
        """ https://www.valuesignals.com/Glossary/Details/Net_Working_Capital?securityId=13381
        """
        return max(0, (self.current_assets - self.excess_cash - (self.current_liabilities - (self.total_debt - self.longterm_debt))))

    @property
    def net_PPE(self):
        try:
            return self.financial_dict[self.ticker]["NetPPE"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def net_fixed_assets(self):
        return self.net_PPE

    @property
    def market_cap(self):
        """ https://www.valuesignals.com/Glossary/Details/Market_Capitalization/13381
        """
        try:
            return self.financial_dict[self.ticker]["marketCap"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def enterprise_value(self):
        """ https://www.valuesignals.com/Glossary/Details/Enterprise_Value
        """
        try:
            return self.financial_dict[self.ticker]["enterpriseValue"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def roc(self):
        return self.ebit / (self.net_working_capital + self.net_fixed_assets)

    @property
    def earnings_yield(self):
        """ https://www.valuesignals.com/Glossary/Details/Earnings_Yield/13381
        """
        return self.ebit / self.enterprise_value

    @property
    def book_market_ratio(self):
        try:
            return 1 / self.financial_dict[self.ticker]["priceToBook"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 0

    @property
    def sector(self):
        try:
            return self.financial_dict[self.ticker]["sector"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 'not available'

    @property
    def financial_date(self):
        try:
            return self.financial_dict[self.ticker]["asOfDate"]
        except Exception as e:
            print(f"Missing information for {self.ticker}\n{e}")
            return 'not available'


def insert_data(conn, ticker_info):
    sql = ''' REPLACE INTO stock_table (ticker, sector, market_cap, roc, earnings_yield, most_recent)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, ticker_info)
    conn.commit()


def update_db(tickers, db_path):
    print("Updating database...")
    conn = sq.connect(db_path, detect_types=sq.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS stock_table")
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_table (
    ticker text PRIMARY KEY,
    sector text,
    market_cap real,
    roc real NOT NULL,
    earnings_yield real NOT NULL,
    most_recent DATE
    );''')

    fs = FinancialStatement(financial_dict)

    for ticker in tickers:

        try:
            fs.set_ticker(ticker)
            data = (ticker, fs.sector, fs.market_cap, fs.roc, fs.earnings_yield, fs.financial_date)
            insert_data(conn, data)

        except Exception as e:
            print(f"Insert data error for ticker {ticker}: {e}. Going to next ticker.")

    if conn:
        conn.close()


def rank_stocks(db_path, csv_path):
    print("Ranking stocks based on Magic Formula...")
    conn = sq.connect(rf'{db_path}')
    cursor = conn.cursor()
    query = cursor.execute(f'''
    SELECT * FROM (
        SELECT *, roc_rank + earnings_yield_rank AS magic_rank FROM
        (
            SELECT *,  RANK ()  OVER( ORDER BY roc DESC) AS roc_rank,
            RANK () OVER( ORDER BY earnings_yield DESC) AS earnings_yield_rank FROM stock_table      
        ) 
    )
    ORDER BY magic_rank ASC
    ''')

    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols, index='ticker')
    df.to_csv(csv_path, index_label='ticker')

    if conn:
        conn.close()


def print_db(db_path):
    conn = sq.connect(rf"{db_path}")
    print(pd.read_sql_query("SELECT * FROM stock_table", conn))
    if conn:
        conn.close()


def dict_to_csv(data: dict, csv_path: Path):
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.sort_index()
    df.to_csv(csv_path, index_label='ticker')


def download_csv_to_df(url: str) -> pd.DataFrame:
    tmp_csv = Path('tmp.csv')

    with tmp_csv.open('wb') as fp:
        content = requests.get(url).content
        fp.write(content)

    df = pd.read_csv(tmp_csv)
    tmp_csv.unlink()
    return df


def chunker(seq: list, size: int) -> Generator:
    return (seq[pos: pos+size] for pos in range(0, len(seq), size))


def _retrieve(chunk_id: int, tickers: list, metric: str, dict_proxy: dict, event: Event) -> int:
    success = 0

    if not event.is_set():
        print(f"Chunk {chunk_id + 1}: Tickers to be retrieved are: {tickers}")
        thread_start = time.time()

        for t in tickers:
            stock = Ticker(t)
            row_dict = dict_proxy.get(t, {k: '' for k in all_keys})
            data = {t: ''}

            if metric == 'financial':
                data = stock.get_financial_data(financial_keys, 'q', trailing=False)

                if isinstance(data, pd.DataFrame):
                    data = data.sort_values(['asOfDate'])
                    data = data.iloc[-1:, :]              # get the latest row
                    data.loc[:, 'asOfDate'] = data.loc[:, 'asOfDate'].astype('str')
                    data = data.to_dict('index')          # dict like {index -> {column -> value}}

            elif metric == 'key_stats':
                data = stock.key_stats

            elif metric == 'price':
                data = stock.price

            elif metric == 'profile':
                data = stock.summary_profile

            if isinstance(data, dict) and isinstance(data[t], dict):
                row_dict.update({k: v for k, v in data[t].items() if k in all_keys})
                dict_proxy[t] = row_dict
                success += 1

        thread_end = time.time()

        print(f"Time elapsed for chunk {chunk_id + 1}: {thread_end - thread_start:.1f} seconds. Metric: {metric} Succeeded: {success}")
        print()

    return success


def get_financial(ticker_list: list, metric: str, keys: list, is_forced: bool) -> dict:
    result = {}

    # Read all saved csv files, renew part.csv first.
    for file in save_root.glob(f"{fn_financial}*.csv"):    # TODO renew part.csv explicitly
        print(f"Loading financial data from {file}...")
        df = pd.read_csv(file, index_col='ticker')
        df = df.sort_index()
        result.update(df.to_dict('index'))

    tickers_need_update = []

    if is_forced:
        tickers_need_update = ticker_list

    else:

        for t in ticker_list:
            row = result.get(t)

            if isinstance(row, dict):

                if all((isinstance(row[k], str) and row[k] == '') or (isinstance(row[k], float) and math.isnan(row[k])) for k in keys):
                    tickers_need_update.append(t)
                    continue

            elif row is None:
                tickers_need_update.append(t)

    if len(tickers_need_update) > 0:
        print(f"Retrieving financial data...")

        with mp.Manager() as manager:
            dict_proxy = manager.dict(result)
            event = Event()    # Once it occurs, all futures stop.

            with concurrent.futures.ThreadPoolExecutor(args.n_thread) as executor:
                chunks = list(chunker(tickers_need_update, args.batch_size))
                tasks = {executor.submit(_retrieve, i, chunk, metric, dict_proxy, event): i
                         for i, chunk in enumerate(chunks)}
                n_banned = 10
                last_n_queue = deque([], n_banned)
                success_counter = 0
                part_csv_path = save_root / f"{fn_financial}_part.csv"

                for future in concurrent.futures.as_completed(tasks.keys()):
                    success = future.result()
                    success_counter += success
                    last_n_queue.append(success)
                    is_banned = len(last_n_queue) == n_banned and all(x == 0 for x in last_n_queue)

                    if is_banned:
                        print(f"\nBanned by yahoo finance API.")
                        event.set()

                    elif (success_counter - 1) % 20 == 0:
                        print(f'\nRetrieved {success_counter} records. Saving file in {part_csv_path}')
                        dict_to_csv(dict(dict_proxy), part_csv_path)

                part_csv_path.unlink(missing_ok=True)
            result.update(dict(dict_proxy))
        dict_to_csv(result, save_root / f"{fn_financial}.csv")
    
    return result


def download_ticker_list(country_code: str) -> list:
    ticker_list = []

    if country_code.upper() == 'US':
        """ US stock data from:
        https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
        ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt
        ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt
        """
        nasdaq_list = 'nasdaqlisted.txt'
        non_nasdaq_list = 'otherlisted.txt'

        try:
            ftp_server = FTP('ftp.nasdaqtrader.com', user='anonymous', passwd='', timeout=5)
            ftp_server.encoding = 'utf-8'
            ftp_server.dir()

        except Exception as e:
            print(f'Fail to connect ftp. {e}')

        else:
            ftp_server.cwd('symboldirectory')

            for file in [nasdaq_list, non_nasdaq_list]:

                with open(file, 'wb') as fp:
                    ftp_server.retrbinary(f"RETR {file}", fp.write)

            ftp_server.quit()

        if Path(nasdaq_list).is_file() and Path(non_nasdaq_list).is_file():
            nasdaq_df = pd.read_csv(nasdaq_list, sep='|')
            non_nasdaq_df = pd.read_csv(non_nasdaq_list, sep='|')
            nasdaq_df = nasdaq_df[(nasdaq_df['ETF'] == 'N') & (nasdaq_df['Test Issue'] == 'N') & (nasdaq_df['Financial Status'] == 'N')]
            non_nasdaq_df = non_nasdaq_df[(non_nasdaq_df['ETF'] == 'N') & (non_nasdaq_df['Test Issue'] == 'N')]
            nasdaq_ticker_list = nasdaq_df['Symbol'].to_list()
            non_nasdaq_ticker_list = non_nasdaq_df['ACT Symbol'].to_list()
            ticker_list = nasdaq_ticker_list + non_nasdaq_ticker_list

    if country_code.upper() == 'TW':
        """ TWSE data from: 
        https://data.gov.tw/datasets/search?p=1&size=10
        上市公司基本資料
        上櫃股票基本資料                
        """
        stock_csv = 'https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv'
        otc_csv = 'https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv'
        csv_urls = [stock_csv, otc_csv]
        suffixes = ['.TW', '.TWO']

        for url, suffix in zip(csv_urls, suffixes):
            df = download_csv_to_df(url)
            series = df['公司代號']
            ticker_list += [f'{ticker}{suffix}' for ticker in series.to_list()]

    return ticker_list


def get_ticker_list(country_code: str) -> list:
    ticker_list_path = save_root / f'{fn_ticker_list}.json'

    if ticker_list_path.is_file():
        print("Loading ticker list...")
        with ticker_list_path.open('rt') as fp:
            ticker_list = json.load(fp)

    else:
        print("Get ticker list...")
        ticker_list = download_ticker_list(country_code)

        with ticker_list_path.open('wt') as fp:
            json.dump(ticker_list, fp, indent=4)
            print(f'{ticker_list_path} is saved.')

    return ticker_list


def remove_outdated(input_dict: dict) -> dict:
    target_date = FiscalDateTime.today().prev_fiscal_quarter.prev_fiscal_quarter.end.strftime('%Y-%m-%d')
    print(f'\nThe end date of the last second fiscal quarter is {target_date}')
    print(f'\nThe financial statement date earlier than {target_date} will be excluded.')

    output_dict = {}

    for ticker, row in input_dict.items():
        if isinstance(row['asOfDate'], str) and row['asOfDate'] >= target_date:
            output_dict[ticker] = row

    return output_dict


def remove_small_marketcap(input_dict: dict) -> dict:
    print(f"\nThe company with market cap < {args.min_market_cap} will be excluded.")
    return {k: v for k, v in input_dict.items() if v['marketCap'] >= args.min_market_cap}


def remove_sector(input_dict: dict) -> dict:
    print(f"\nThe company sector in {exclude_sectors} will be excluded.")
    return {k: v for k, v in input_dict.items() if v['sector'] not in exclude_sectors}


def process_args():
    parser = argparse.ArgumentParser(description='Process refresh options')
    parser.add_argument('--country', '-c', type=str, default='US', dest='country',
                        help='Get stocks from which country')
    parser.add_argument('--batch-size', '-b', type=int, default=1, dest='batch_size',
                        help='Number of tickers in a batch')
    parser.add_argument('--thread', '-t', type=int, default=1, dest='n_thread',
                        help='Number of threads in parallel')
    parser.add_argument('--price', action='store_true', dest='force_price',
                        help='Renew price')
    parser.add_argument('--key-stats', action='store_true', dest='force_key_stats',
                        help='Renew key stats')
    parser.add_argument('--financial', action='store_true', dest='force_financial',
                        help='Renew financial statement')
    parser.add_argument('--profile', action='store_true', dest='force_profile',
                        help='Renew summary profile')
    parser.add_argument('--min-market-cap', '-m', type=int, default=1e9, dest='min_market_cap',
                        help='Minimal market cap')
    return parser.parse_args()


def main():
    global financial_dict

    # Retrieve or load ticker list
    ticker_list = get_ticker_list(args.country)

    # Load old financial data then update it
    for metric, keys, is_forced in zip(metric_list, keys_list, force_renew_list):
        financial_dict = get_financial(ticker_list, metric, keys, is_forced)

    financial_dict = remove_outdated(financial_dict)
    financial_dict = remove_small_marketcap(financial_dict)
    financial_dict = remove_sector(financial_dict)
    update_db(financial_dict.keys(), save_root / f"{fn_stock_rank}.db")
    rank_stocks(save_root / f"{fn_stock_rank}.db", save_root / f"{fn_stock_rank}.csv")


if __name__ == '__main__':
    args = process_args()
    save_root = Path(args.country.upper())
    save_root.mkdir(0o755, exist_ok=True)

    financial_dict = {}

    fn_ticker_list = 'ticker_list'
    fn_financial = 'financial'
    fn_stock_rank = 'stock_rank'

    financial_keys = ["asOfDate", "currencyCode", "TotalDebt", "LongTermDebt", "CurrentAssets", "CurrentLiabilities",
                      "NetPPE", "EBIT", "CashCashEquivalentsAndShortTermInvestments"]
    key_stats_keys = ["enterpriseValue", "priceToBook"]
    price_keys = ["marketCap"]
    profile_keys = ["sector"]
    all_keys = profile_keys + price_keys + financial_keys + key_stats_keys

    metric_list = ["financial", "profile", "price", "key_stats"]
    keys_list = [financial_keys, profile_keys, price_keys, key_stats_keys]
    force_renew_list = [args.force_financial, args.force_profile, args.force_price, args.force_key_stats]

    exclude_sectors = ["Financial Services", "Utilities"]

    start = time.time()
    main()
    end = time.time()

    print(f"Execution time: {end - start:.1f} seconds")
