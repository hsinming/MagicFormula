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
    def __init__(self, sheet: dict):
        self.ticker = ''
        self.sheet = sheet

    def set_ticker(self, ticker: str):
        self.ticker = ticker

    @property
    def ebit(self):
        return self.sheet[self.ticker]["EBIT"]

    @property
    def total_assets(self):
        return self.sheet[self.ticker]["TotalAssets"]

    @property
    def total_debt(self):
        return self.sheet[self.ticker]["TotalDebt"]

    @property
    def current_assets(self):
        result = self.sheet[self.ticker]["CurrentAssets"]
        if math.isnan(result):
            print(f"Missing current assets for {self.ticker}")
            result = 0
        return result

    @property
    def current_liabilities(self):
        result = self.sheet[self.ticker]["CurrentLiabilities"]
        if math.isnan(result):
            print(f"Missing current liabilities for {self.ticker}")
            result = 0
        return result

    @property
    def longterm_debt(self):
        result = self.sheet[self.ticker]["LongTermDebt"]
        if math.isnan(result):
            print(f"Missing longterm debt for {self.ticker}")
            result = 0
        return result

    @property
    def intangible_assets(self):
        result = self.sheet[self.ticker]["GoodwillAndOtherIntangibleAssets"]
        if math.isnan(result):
            print(f"Missing intangible assets for {self.ticker}")
            result = 0
        return result

    @property
    def total_cash(self):
        result = self.sheet[self.ticker]["CashCashEquivalentsAndShortTermInvestments"]
        if math.isnan(result):
            print(f"Missing total cash for {self.ticker}")
            result = 0
        return result

    @property
    def excess_cash(self):
        """ There are many definitions of excess cash.
        definition 1 : https://www.valupaedia.com/index.php/business-dictionary/552-excess-cash
        excess_cash = self.total_cash - max(0, self.current_liabilities - (self.current_assets - self.total_cash))
        definition 2: https://www.quant-investing.com/glossary/excess-cash
        excess_cash = min(self.total_cash, max(self.current_assets - 2.0 * self.current_liabilities, 0))
        """
        return self.total_cash - max(0, self.current_liabilities - (self.current_assets - self.total_cash))

    @property
    def net_property_plant_equipment(self):
        try:
            return self.sheet[self.ticker]["NetPPE"]
        except Exception as e:
            print(f"Missing net PPE for {self.ticker}\n{e}")
            return 0

    @property
    def net_fixed_assets(self):
        """
        definition 1: https://www.valuesignals.com/Glossary/Details/Net_Fixed_Assets/13381
        Net fixed assets = net PPE
        definition 2: https://www.quant-investing.com/glossary/net-fixed-assets
        Net Fixed Assets = Total Assets - Total Current Assets - Total Intangible assets
        """
        return self.net_property_plant_equipment

    @property
    def net_working_capital(self):
        """ https://www.valuesignals.com/Glossary/Details/Net_Working_Capital?securityId=13381
        """
        return max(0, (self.current_assets - self.excess_cash - (
                    self.current_liabilities - (self.total_debt - self.longterm_debt))))

    @property
    def market_cap(self):
        """ https://www.valuesignals.com/Glossary/Details/Market_Capitalization/13381
        """
        return self.sheet[self.ticker]["marketCap"]

    @property
    def enterprise_value(self):
        """
        definition 1: https://www.quant-investing.com/glossary/enterprise-value
        EV = market cap + long-term debt + minority interest + preferred stock - excess cash
        definition 2: https://www.valuesignals.com/Glossary/Details/Enterprise_Value/13381
        EV = market cap + total debt + minority interest + preferred stock - total cash
        """
        return self.market_cap + self.longterm_debt - self.excess_cash

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
        return self.sheet[self.ticker]["bookValue"] / self.sheet[self.ticker]["regularMarketPrice"]

    @property
    def sector(self):
        return self.sheet[self.ticker]["sector"]

    @property
    def financial_date(self):
        return self.sheet[self.ticker]["asOfDate"]

    @property
    def country(self):
        return self.sheet[self.ticker]["country"]

    @property
    def name(self):
        return self.sheet[self.ticker]["longName"]


def insert_data(conn, ticker_info):
    sql = ''' REPLACE INTO stock_table (ticker, name, most_recent, roc, earnings_yield)
              VALUES(?,?,?,?,?) '''
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
    name text,
    most_recent DATE,    
    roc real NOT NULL,
    earnings_yield real NOT NULL    
    );''')

    fs = FinancialStatement(financial_dict)

    for ticker in tickers:

        try:
            fs.set_ticker(ticker)
            data = (ticker, fs.name, fs.financial_date, fs.roc, fs.earnings_yield)
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


def dict_to_csv(input_dict: dict, csv_path: Path):
    df = pd.DataFrame.from_dict(input_dict, orient='index')
    df = df.sort_index()
    df.to_csv(csv_path, index_label='ticker', encoding='utf-8')


def renew_column(input_dict: dict) -> dict:
    result = {}

    for ticker, row in input_dict.items():
        new_row = {k: math.nan for k in all_keys}

        for k, v in row.items():

            if k in all_keys:
                new_row[k] = v

        result[ticker] = new_row

    return result


def chunker(seq: list, size: int) -> Generator:
    return (seq[pos: pos+size] for pos in range(0, len(seq), size))


def is_complete(input_dict: dict, keys_to_check: list, min: int) -> bool:
    check_list = []

    for k in keys_to_check:

        if input_dict.get(k) is not None:

            if isinstance(input_dict[k], str) and input_dict[k] != '':
                check_list.append(True)

            if isinstance(input_dict[k], float) and not math.isnan(input_dict[k]):
                check_list.append(True)

        else:
            check_list.append(False)

    return check_list.count(True) >= min


def _retrieve(chunk_id: int, tickers: list, metric: str, dict_proxy: dict, event: Event) -> int:
    success = 0

    if not event.is_set():
        print(f"Chunk {chunk_id + 1}: Tickers to be retrieved are: {tickers}")
        thread_start = time.time()

        for t in tickers:
            stock = Ticker(t, country=country_code[args.country.upper()])
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

    # Loading order is important: load part csv in the last because it is usually newer.
    for file in sorted(save_root.glob(f'{fn_financial}*.csv')):
        print(f"Loading financial data from {file}...")
        df = pd.read_csv(file, index_col='ticker')
        df = df.sort_index()
        old_dict = df.to_dict('index')
        new_dict = renew_column(old_dict)
        result.update(new_dict)

    tickers_need_update = []

    if is_forced:
        tickers_need_update = ticker_list

    else:

        for t in ticker_list:
            row = result.get(t)

            if isinstance(row, dict):

                if not is_complete(row, keys, 1):
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
                tasks = [executor.submit(_retrieve, i, chunk, metric, dict_proxy, event) for i, chunk in enumerate(chunks)]
                n_banned = 10
                last_n_queue = deque([], n_banned)
                success_counter = 0
                part_csv_path = save_root / f"{fn_financial}_part.csv"

                for future in concurrent.futures.as_completed(tasks):
                    success = future.result()
                    success_counter += success
                    last_n_queue.append(success)
                    is_banned = last_n_queue.count(0) == n_banned

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


def download_csv_to_df(url: str) -> pd.DataFrame:
    tmp_csv = Path('tmp.csv')

    with tmp_csv.open('wb') as fp:
        content = requests.get(url).content
        fp.write(content)

    df = pd.read_csv(tmp_csv)
    tmp_csv.unlink()
    return df


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

        if Path(nasdaq_list).is_file():
            nasdaq_df = pd.read_csv(nasdaq_list, sep='|')
            nasdaq_df = nasdaq_df[(nasdaq_df['ETF'] == 'N') & (nasdaq_df['Test Issue'] == 'N') & (nasdaq_df['Financial Status'] == 'N')]
            nasdaq_ticker_list = nasdaq_df['Symbol'].to_list()
            ticker_list += nasdaq_ticker_list

        if Path(non_nasdaq_list).is_file():
            non_nasdaq_df = pd.read_csv(non_nasdaq_list, sep='|')
            non_nasdaq_df = non_nasdaq_df[(non_nasdaq_df['ETF'] == 'N') & (non_nasdaq_df['Test Issue'] == 'N')]
            non_nasdaq_ticker_list = non_nasdaq_df['ACT Symbol'].to_list()
            ticker_list += non_nasdaq_ticker_list

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

    ticker_list = [t for t in ticker_list if "$" not in t]

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
    result = {}

    for k, v in input_dict.items():
        try:
            market_cap = float(v['marketCap'])
        except:
            continue

        if market_cap >= args.min_market_cap:
            result[k] = v
            result[k]['marketCap'] = float(result[k]['marketCap'])

    return result


def remove_sector(input_dict: dict) -> dict:
    print(f"\nThe company sector in {excluded_sectors} will be excluded.")
    return {k: v for k, v in input_dict.items() if v['sector'] not in excluded_sectors}


def remove_country(input_dict: dict) -> dict:
    accepted_country = country_code[args.country.upper()]
    print(f"\nThe company country not {accepted_country} will be excluded.")
    return {k: v for k, v in input_dict.items() if v['country'] == accepted_country}


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
    parser.add_argument('--min-market-cap', '-m', type=int, default=1e8, dest='min_market_cap',
                        help='Minimal market cap')
    return parser.parse_args()


def main():
    global financial_dict

    # Retrieve or load ticker list
    ticker_list = get_ticker_list(args.country)

    # Load old financial data then update it
    for metric, keys, is_forced in zip(metric_list, keys_list, force_renew_list):
        financial_dict = get_financial(ticker_list, metric, keys, is_forced)

    # Exclude inappropriate tickers
    for filter in filter_list:
        financial_dict = filter(financial_dict)

    update_db(financial_dict.keys(), save_root / f"{fn_stock_rank}.db")
    rank_stocks(save_root / f"{fn_stock_rank}.db", save_root / f"{fn_stock_rank}.csv")


if __name__ == '__main__':
    args = process_args()
    save_root = Path(args.country.upper())
    save_root.mkdir(0o755, exist_ok=True)

    financial_dict = {}
    country_code = {'US': 'United States', 'TW': 'Taiwan'}
    fn_ticker_list = 'ticker_list'
    fn_financial = 'financial'
    fn_stock_rank = 'stock_rank'

    financial_keys = ["asOfDate", "EBIT", "TotalAssets", "TotalDebt", "LongTermDebt", "CurrentAssets", "CurrentLiabilities",
                      "GoodwillAndOtherIntangibleAssets", "NetPPE", "CashCashEquivalentsAndShortTermInvestments"]
    key_stats_keys = ["bookValue"]
    price_keys = ["longName", "marketCap", "currency", "regularMarketPrice"]
    profile_keys = ["sector", "country"]
    all_keys = profile_keys + price_keys + financial_keys + key_stats_keys

    metric_list = ["financial", "profile", "price", "key_stats"]
    keys_list = [financial_keys, profile_keys, price_keys, key_stats_keys]
    force_renew_list = [args.force_financial, args.force_profile, args.force_price, args.force_key_stats]
    excluded_sectors = ["Financial Services", "Utilities"]
    filter_list = [remove_outdated, remove_sector, remove_small_marketcap, remove_country]

    start = time.time()
    main()
    end = time.time()

    print(f"Execution time: {end - start:.1f} seconds")
