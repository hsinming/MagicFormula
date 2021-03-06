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
from datetime import datetime, timedelta
from pathlib import Path
from ftplib import FTP
import sqlite3 as sq
from itertools import chain
import pandas as pd
import requests
from yahooquery import Ticker
from fiscalyear import FiscalDate
import twder
from currency_converter import CurrencyConverter, SINGLE_DAY_ECB_URL


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
        result = self.sheet[self.ticker]['LongTermDebtAndCapitalLeaseObligation']
        if math.isnan(result):
            print(f"Missing longterm debt for {self.ticker}")
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
    def most_recent_quarter(self):
        return self.sheet[self.ticker]["asOfDate"]

    @property
    def price_date(self):
        timestamp = self.sheet[self.ticker]["regularMarketTime"]
        dt_obj = datetime.fromtimestamp(timestamp)
        dt_str = dt_obj.strftime('%Y-%m-%d')
        return dt_str

    @property
    def country(self):
        return self.sheet[self.ticker]["country"]

    @property
    def name(self):
        return self.sheet[self.ticker]["longName"]


def insert_data(conn, ticker_info):
    sql = ''' REPLACE INTO stock_table (ticker, name, sector, most_recent_quarter, price_from, roc, earnings_yield, book_market_ratio)
              VALUES(?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, ticker_info)
    conn.commit()


def update_db(financial_dict, db_path):
    print("Updating database...")
    conn = sq.connect(db_path, detect_types=sq.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS stock_table")
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_table (
    ticker TEXT PRIMARY KEY NOT NULL,
    name TEXT,
    sector TEXT,
    most_recent_quarter DATE,
    price_from DATE,
    roc REAL NOT NULL,
    earnings_yield REAL NOT NULL,
    book_market_ratio REAL
    );''')

    fs = FinancialStatement(financial_dict)

    for ticker in financial_dict.keys():

        try:
            fs.set_ticker(ticker)
            data = (ticker, fs.name, fs.sector, fs.most_recent_quarter, fs.price_date, fs.roc, fs.earnings_yield, fs.book_market_ratio)
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
        SELECT *, roc_rank + earnings_yield_rank AS magic_rank 
        FROM
        (
            SELECT *,  
            RANK () OVER( ORDER BY roc DESC) AS roc_rank,
            RANK () OVER( ORDER BY earnings_yield DESC) AS earnings_yield_rank 
            FROM stock_table
        )
    )
    WHERE book_market_ratio > 0
    ORDER BY magic_rank ASC
    ''')

    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols, index='ticker')
    df.to_csv(csv_path, index_label='ticker')

    if conn:
        conn.close()


def dict_to_csv(input_dict: dict, csv_path: Path):
    df = pd.DataFrame.from_dict(input_dict, orient='index')
    df = df.sort_index()
    df.to_csv(csv_path, index_label='ticker', encoding='utf-8')


def change_to_new_keys(input_dict: dict) -> dict:
    result = {}

    for ticker, old_row in input_dict.items():
        new_row = {k: math.nan for k in all_keys}

        for k, v in old_row.items():

            if k in all_keys:
                new_row[k] = v

        result[ticker] = new_row

    return result


def is_complete(input_dict: dict, keys_to_check: list, min_required: int) -> bool:
    check_list = []

    for k in keys_to_check:

        if input_dict.get(k) is not None:

            if isinstance(input_dict[k], str) and input_dict[k] != '':
                check_list.append(True)

            if isinstance(input_dict[k], float) and not math.isnan(input_dict[k]):
                check_list.append(True)

        else:
            check_list.append(False)

    return check_list.count(True) >= min_required


def get_financial(ticker_list: list, metric: str, keys: list, is_forced: bool) -> dict:
    result = {}
    tickers_to_pull = []

    # Loading order is important: load part csv in the last because it is usually newer.
    for file in sorted(save_root.glob(f'{fn_financial}*.csv')):
        print(f"Loading financial data from {file}...")
        df = pd.read_csv(file, index_col='ticker')
        df = df.sort_index()
        old_dict = df.to_dict('index')
        new_dict = change_to_new_keys(old_dict)
        result.update(new_dict)

    # keep only those tickers in ticker_list
    result = {k: v for k, v in result.items() if k in ticker_list}

    if is_forced:
        tickers_to_pull = ticker_list

    else:

        for t in ticker_list:

            if t not in result.keys() or not is_complete(result[t], keys, 1):
                tickers_to_pull.append(t)

    if len(tickers_to_pull) > 0:
        print(f"Pulling {len(tickers_to_pull)} tickers {metric}...")
        pulled_counter = 0
        part_csv_path = save_root / f"{fn_financial}_part.csv"

        for t in tickers_to_pull:
            print(f"Pulling {t} {metric}...")
            stock = Ticker(t, country=yahoo_country)
            row_dict = result.get(t, {k: math.nan for k in all_keys})
            data = None

            if metric == 'financial':
                pulled_data = stock.get_financial_data(keys, 'q', trailing=True)

                if isinstance(pulled_data, pd.DataFrame):
                    ttm_df = pulled_data[pulled_data['periodType'] == 'TTM']
                    ttm_df = ttm_df.sort_values(['asOfDate'])
                    ttm_df = ttm_df.iloc[-1:, :]                # get the latest row
                    quarterly_df = pulled_data[pulled_data['periodType'] == '3M']
                    quarterly_df = quarterly_df.sort_values(['asOfDate'])
                    quarterly_df = quarterly_df.iloc[-1:, :]    # get the latest row

                    if len(quarterly_df) == len(ttm_df) == 1:
                        result_df = quarterly_df
                        result_df.iat[0, result_df.columns.get_loc('EBIT')] = ttm_df.iat[0, ttm_df.columns.get_loc('EBIT')]
                        result_df = result_df.astype({'asOfDate': 'str'})
                        data = result_df.to_dict('index')        # a nested dict like {index -> {column -> value}}

            elif metric == 'quotes':
                data = stock.quotes

            elif metric == 'profile':
                data = stock.summary_profile

            if isinstance(data, dict) and isinstance(data.get(t), dict):
                new_row = {k: v for k, v in data[t].items() if k in all_keys}
                [print(f"\t{k} -> {v}") for k, v in new_row.items()]
                row_dict.update(new_row)
                result[t] = row_dict
                pulled_counter += 1

            if (pulled_counter + 1) % 20 == 0:
                print()
                print(f"Saving file in {part_csv_path}")
                dict_to_csv(result, part_csv_path)

            print()
            time.sleep(3)    # avoid yahoo api ban

        part_csv_path.unlink(missing_ok=True)

    print(f"Saving file in {save_root / f'{fn_financial}.csv'}")
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
            mask1 = (nasdaq_df['Market Category'].isin(['Q', 'G']))  # Q=NASDAQ Global Select, G=NASDAQ Global
            mask2 = (nasdaq_df['Test Issue'] == 'N')
            mask3 = (nasdaq_df['Financial Status'] == 'N')
            mask4 = (nasdaq_df['ETF'] == 'N')
            mask5 = ~(nasdaq_df['Security Name'].str.contains('Unit|Warrant|Right|Preferred|Convertible', case=True, na=False))
            nasdaq_df = nasdaq_df[mask1 & mask2 & mask3 & mask4 & mask5]
            nasdaq_ticker_list = nasdaq_df['Symbol'].to_list()
            ticker_list += nasdaq_ticker_list

        if Path(non_nasdaq_list).is_file():
            non_nasdaq_df = pd.read_csv(non_nasdaq_list, sep='|')
            mask6 = (non_nasdaq_df['Exchange'].isin(['N', 'A']))  # N=NYSE  A=NYSE MKT(AMEX)
            mask7 = (non_nasdaq_df['ETF'] == 'N')
            mask8 = (non_nasdaq_df['Test Issue'] == 'N')
            mask9 = ~(non_nasdaq_df['Security Name'].str.contains('Unit|Warrant|Right|Preferred|Convertible', case=True, na=False))
            mask10 = ~(non_nasdaq_df['ACT Symbol'].str.contains('\.U|\.W|\.R|\$|\.D|\.Z|\.V', case=True, na=False))
            non_nasdaq_df = non_nasdaq_df[mask6 & mask7 & mask8 & mask9 & mask10]
            non_nasdaq_ticker_list = non_nasdaq_df['ACT Symbol'].to_list()
            ticker_list += non_nasdaq_ticker_list

    if country_code.upper() == 'TW':
        """ TWSE data from:
        https://data.gov.tw/datasets/search?p=1&size=10
        key words = ["????????????????????????", "????????????????????????"]
        """
        stock_csv = 'https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv'
        otc_csv = 'https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv'
        csv_urls = [stock_csv, otc_csv]
        suffixes = ['.TW', '.TWO']

        for url, suffix in zip(csv_urls, suffixes):
            df = download_csv_to_df(url)
            series = df['????????????']
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
    tmp_date = datetime.today() - timedelta(days=45)
    target_date = FiscalDate(tmp_date.year, tmp_date.month, tmp_date.day).prev_fiscal_quarter.end.strftime('%Y-%m-%d')
    print(f'\nThe financial statement date earlier than {target_date} will be excluded.')

    output_dict = {}

    for k, v in input_dict.items():
        if isinstance(v['asOfDate'], str) and v['asOfDate'] >= target_date:
            output_dict[k] = v

    return output_dict


def remove_small_marketcap(input_dict: dict) -> dict:
    print(f"\nThe company with market cap < {args.min_market_cap:,.0f} USD will be excluded.")
    usd_converter = CurrencyConverter(SINGLE_DAY_ECB_URL)
    twd_converter = float(twder.now('USD')[1])
    result = {}

    for k, v in input_dict.items():

        try:
            market_cap = float(v['marketCap'])
            currency = str(v['currency']).upper()

        except Exception:
            continue

        if not math.isnan(market_cap):

            if currency == 'TWD':
                market_cap_in_usd = market_cap / twd_converter

            else:
                market_cap_in_usd = usd_converter.convert(market_cap, currency, 'USD')  # TWD not included.

            if market_cap_in_usd >= args.min_market_cap:
                result[k] = v
                result[k]['marketCap'] = market_cap

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
    parser.add_argument('--min-market-cap', '-m', type=int, default=1e9, dest='min_market_cap',
                        help='Minimal market cap in USD')
    parser.add_argument('--quotes', action='store_true', dest='force_quotes',
                        help='Renew quotes')
    parser.add_argument('--financial', action='store_true', dest='force_financial',
                        help='Renew financial statement')
    parser.add_argument('--profile', action='store_true', dest='force_profile',
                        help='Renew summary profile')

    return parser.parse_args()


def main():
    # Retrieve or load ticker list
    ticker_list = get_ticker_list(args.country)

    # Load old financial data then update it
    financial_dict = {}

    for metric, keys, is_forced in zip(metric_list, keys_list, force_renew_list):
        financial_dict = get_financial(ticker_list, metric, keys, is_forced)

    # Exclude inappropriate tickers
    for f in filter_list:
        financial_dict = f(financial_dict)

    update_db(financial_dict, save_root / f"{fn_stock_rank}.db")
    rank_stocks(save_root / f"{fn_stock_rank}.db", save_root / f"{fn_stock_rank}.csv")


if __name__ == '__main__':
    args = process_args()
    save_root = Path(args.country.upper())
    save_root.mkdir(0o755, exist_ok=True)

    fn_ticker_list = 'ticker_list'
    fn_financial = 'financial'
    fn_stock_rank = 'stock_rank'
    country_code = {'US': 'United States', 'TW': 'Taiwan'}
    yahoo_country = country_code[args.country.upper()]

    financial_keys = ["asOfDate", "EBIT", "TotalAssets", "TotalDebt", 'LongTermDebtAndCapitalLeaseObligation',
                      "CurrentAssets", "CurrentLiabilities", "NetPPE", "CashCashEquivalentsAndShortTermInvestments"]
    profile_keys = ["sector", "country"]
    quotes_keys = ["longName", "currency", "marketCap", "bookValue", "regularMarketPrice", "regularMarketTime"]

    metric_list = ["profile", "quotes", "financial"]
    keys_list = [profile_keys, quotes_keys, financial_keys]
    force_renew_list = [args.force_profile, args.force_quotes, args.force_financial]
    excluded_sectors = ["Financial Services", "Financial", "Utilities", "Real Estate"]
    filter_list = [remove_outdated, remove_sector, remove_small_marketcap, remove_country]
    all_keys = list(chain.from_iterable(keys_list))

    start = time.time()
    main()
    end = time.time()

    print(f"Execution time: {end - start:.1f} seconds")
