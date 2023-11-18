#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@author: Hsin-ming Chen
@license: GPL
@file: main.py
@time: 2023/08/14
@contact: hsinming.chen@gmail.com
@software: PyCharm
"""
import json
import argparse
import math
import time
import sqlite3 as sq
from datetime import datetime, timedelta
from pathlib import Path
from ftplib import FTP
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

    def _get_number(self, key: str) -> float:
        result = self.sheet[self.ticker][key]
        if isinstance(result, str):
            result = float(result)
        if math.isnan(result):
            print(f"Missing {key} for {self.ticker}. Set to zero.")
            result = 0.0
        return result

    @property
    def ebit_ttm(self):
        return self._get_number('EBIT')

    @property
    def total_assets(self):
        return self._get_number('TotalAssets')

    @property
    def total_debt(self):
        return self._get_number('TotalDebt')

    @property
    def current_assets(self):
        return self._get_number('CurrentAssets')

    @property
    def current_liabilities(self):
        return self._get_number('CurrentLiabilities')

    @property
    def longterm_debt(self):
        return self._get_number('LongTermDebtAndCapitalLeaseObligation')

    @property
    def minority_interest(self):
        return self._get_number('MinorityInterest')

    @property
    def preferred_stock(self):
        return self._get_number('PreferredStock')

    @property
    def total_cash(self):
        """
        https://www.valupaedia.com/index.php/business-dictionary/552-excess-cash
        Total Cash = Cash and cash equivalents + short term investments
        """
        return self._get_number('CashCashEquivalentsAndShortTermInvestments')

    @property
    def excess_cash(self):
        """ There are many definitions of excess cash.
        definition 1 : https://www.valupaedia.com/index.php/business-dictionary/552-excess-cash
        excess_cash = total_cash - max(0, current_liabilities - (current_assets - total_cash))

        definition 2: https://www.quant-investing.com/glossary/excess-cash
        excess_cash = min(total_cash, max(0, current_assets - 2.0 * current_liabilities))
        """
        return self.total_cash - max(0, self.current_liabilities - (self.current_assets - self.total_cash))

    @property
    def net_property_plant_equipment(self):
        return self._get_number('NetPPE')

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
        """
        https://www.valuesignals.com/Glossary/Details/Net_Working_Capital?securityId=13381
        Net Working Capital = MAX(0, Current Asset - Excess Cash - (Current Liability - (Total Debt - Long Term Debt)))
        """
        return max(0, (self.current_assets - self.excess_cash - (self.current_liabilities - (self.total_debt - self.longterm_debt))))

    @property
    def market_cap(self):
        """
        https://www.valuesignals.com/Glossary/Details/Market_Capitalization/13381
        market cap = share price * common share outstanding
        """
        return self._get_number('marketCap')

    @property
    def enterprise_value(self):
        """
        definition 1: https://www.quant-investing.com/glossary/enterprise-value
        Enterprise value = market cap + long-term debt + minority interest + preferred stock - excess cash

        definition 2: https://www.valuesignals.com/Glossary/Details/Enterprise_Value/13381
        Enterprise value = market cap + total debt + minority interest + preferred stock - total cash
        """
        return self.market_cap + self.longterm_debt + self.minority_interest + self.preferred_stock - self.excess_cash

    @property
    def roc(self):
        """
        https://www.valuesignals.com/Glossary/Details/ROC
        ROC = EBIT_TTM / (Net Working Capital + Net Fixed Assets)
        """
        return self.ebit_ttm / (self.net_working_capital + self.net_fixed_assets)

    @property
    def earnings_yield(self):
        """
        https://www.valuesignals.com/Glossary/Details/Earnings_Yield/13381
        earnings yield = EBIT_TTM / Enterprise Value
        """
        return self.ebit_ttm / self.enterprise_value

    @property
    def book_value(self):
        return self._get_number('bookValue')

    @property
    def regular_market_price(self):
        price = self._get_number('regularMarketPrice') if self._get_number('regularMarketPrice') > 0 else -1
        return price

    @property
    def book_market_ratio(self):
        return self.book_value / self.regular_market_price

    @property
    def sector(self):
        return self.sheet[self.ticker]["sector"]

    @property
    def most_recent_quarter(self):
        return self.sheet[self.ticker]["asOfDate"]

    @property
    def price_date(self):
        market_date_time = self.sheet[self.ticker]["regularMarketTime"]
        dt_obj = datetime.fromisoformat(market_date_time)
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
        fs.set_ticker(ticker)
        data = (ticker, fs.name, fs.sector, fs.most_recent_quarter, fs.price_date, fs.roc, fs.earnings_yield,
                fs.book_market_ratio)
        insert_data(conn, data)
        # try:
        #     data = (ticker, fs.name, fs.sector, fs.most_recent_quarter, fs.price_date, fs.roc, fs.earnings_yield, fs.book_market_ratio)
        # except Exception as e:
        #     print(f"Insert data error for ticker {ticker}: {e}. Going to next ticker.")
        #     continue
        # else:
        #     insert_data(conn, data)

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

    # Keep only those tickers in ticker_list
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
            stock = Ticker(t, country=country_dict[args.country.upper()])
            row_dict = result.get(t, {k: math.nan for k in all_keys})
            data = None

            try:
                if metric == 'financial':
                    pulled_data = stock.get_financial_data(keys, 'q', trailing=True)

                    if isinstance(pulled_data, pd.DataFrame):
                        ttm_df = pulled_data[pulled_data['periodType'] == 'TTM']
                        ttm_df = ttm_df.sort_values(['asOfDate'])
                        ttm_df = ttm_df.iloc[-1:, :]                # get the latest TTM
                        quarterly_df = pulled_data[pulled_data['periodType'] == '3M']
                        quarterly_df = quarterly_df.sort_values(['asOfDate'])
                        quarterly_df = quarterly_df.iloc[-1:, :]    # get the latest quarter

                        if len(quarterly_df) == len(ttm_df) == 1:
                            result_df = quarterly_df                # financial data comes mainly from quarterly_df

                            # replace quarterly EBIT with EBIT(TTM)
                            result_df.iat[0, result_df.columns.get_loc('EBIT')] = ttm_df.iat[0, ttm_df.columns.get_loc('EBIT')]

                            # convert DateTime type to string to save in dict
                            result_df = result_df.astype({'asOfDate': 'str'})
                            data = result_df.to_dict('index')       # a nested dict like {index -> {column -> value}}

                elif metric == 'price':
                    data = stock.price

                elif metric == 'stat':
                    data = stock.key_stats

                elif metric == 'profile':
                    data = stock.summary_profile

            except Exception as e:
                print(e)
                continue

            # Renew row by row
            if isinstance(data, dict) and isinstance(data.get(t), dict):
                new_row = {k: v for k, v in data[t].items() if k in all_keys}
                [print(f"\t{k} -> {v}") for k, v in new_row.items()]
                row_dict.update(new_row)
                result[t] = row_dict
                pulled_counter += 1

            # Save csv every 20 successful retrival
            if (pulled_counter + 1) % 20 == 0:
                print()
                print(f"Saving file in {part_csv_path}")
                dict_to_csv(result, part_csv_path)

            print()
            time.sleep(3)    # avoid ban by yahoo api

        part_csv_path.unlink(missing_ok=True)

    print(f"Saving file in {save_root / f'{fn_financial}.csv'}")
    dict_to_csv(result, save_root / f"{fn_financial}.csv")

    return result


def download_csv_to_df(url: str) -> pd.DataFrame:
    csv_path = save_root / url.split('/')[-1]

    with csv_path.open('wb') as fp:
        content = requests.get(url).content
        fp.write(content)

    df = pd.read_csv(csv_path)
    return df


def download_ticker_list(country_code: str) -> list:
    ticker_list = []

    if country_code.upper() == 'US':
        """ Get US stock tickers from:
        https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
        ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt
        ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt
        """
        nasdaq_security_path = save_root / 'nasdaqlisted.txt'
        other_security_path = save_root / 'otherlisted.txt'

        try:
            ftp_server = FTP('ftp.nasdaqtrader.com', user='anonymous', passwd='', timeout=5)
            ftp_server.encoding = 'utf-8'
            ftp_server.dir()
        except Exception as e:
            print(f'Fail to connect ftp. {e}')
        else:
            ftp_server.cwd('symboldirectory')
            for file, save_path in [('nasdaqlisted.txt', nasdaq_security_path), ('otherlisted.txt', other_security_path)]:
                with save_path.open('wb') as fp:
                    ftp_server.retrbinary(f"RETR {file}", fp.write)
            ftp_server.quit()

        if Path(nasdaq_security_path).is_file():
            nasdaq_df = pd.read_csv(nasdaq_security_path, sep='|')
            mask1 = (nasdaq_df['Market Category'].isin(['Q', 'G']))  # Q=NASDAQ Global Select, G=NASDAQ Global
            mask2 = (nasdaq_df['Test Issue'] == 'N')
            mask3 = (nasdaq_df['Financial Status'] == 'N')
            mask4 = (nasdaq_df['ETF'] == 'N')
            nasdaq_bad_keywords = ['Preferred', 'Convertible', 'Fund', 'Notes', 'Debentures', 'Depositary', 'Warrant',
                                   'ADR', 'ADS', 'ETF', 'ETN', '%', '\$', '- Right', '- Unit', '- Subunit']
            mask5 = ~(nasdaq_df['Security Name'].str.contains('|'.join(nasdaq_bad_keywords), case=True, na=False))
            nasdaq_df = nasdaq_df[mask1 & mask2 & mask3 & mask4 & mask5]
            nasdaq_ticker_list = nasdaq_df['Symbol'].to_list()
            ticker_list += nasdaq_ticker_list

        if Path(other_security_path).is_file():
            non_nasdaq_df = pd.read_csv(other_security_path, sep='|')
            mask6 = (non_nasdaq_df['Exchange'].isin(['N', 'A']))  # N=NYSE  A=NYSE MKT(AMEX)
            mask7 = (non_nasdaq_df['ETF'] == 'N')
            mask8 = (non_nasdaq_df['Test Issue'] == 'N')
            other_bad_keywords = ['Preferred', 'Convertible', 'Fund', 'Notes', 'Debentures', 'Depositary', 'Warrant',
                                  'ADR', 'ADS', 'ETF', 'ETN', '%', '\$']
            mask9 = ~(non_nasdaq_df['Security Name'].str.contains('|'.join(other_bad_keywords), case=True, na=False))
            mask10 = ~(non_nasdaq_df['ACT Symbol'].str.contains('\.U|\.W|\.R|\.D|\.Z|\.V|\$', case=True, na=False))
            non_nasdaq_df = non_nasdaq_df[mask6 & mask7 & mask8 & mask9 & mask10]
            non_nasdaq_ticker_list = non_nasdaq_df['ACT Symbol'].to_list()
            non_nasdaq_ticker_list = [t.replace('.', '-') for t in non_nasdaq_ticker_list]    # Yahoo Finance's naming rule
            ticker_list += non_nasdaq_ticker_list

    if country_code.upper() == 'TW':
        """ Get TWSE tickers from:
        https://data.gov.tw/datasets/search?p=1&size=10
        search key words = ["上市公司基本資料", "上櫃股票基本資料"]
        上市公司基本資料 https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv
        上櫃股票基本資料 https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv
        """
        stock_csv = 'https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv'
        otc_csv = 'https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv'
        csv_urls = [stock_csv, otc_csv]
        suffixes = ['.TW', '.TWO']    # for Yahoo Finance's naming rule

        for url, suffix in zip(csv_urls, suffixes):
            df = download_csv_to_df(url)
            series = df['公司代號']
            ticker_list += [f'{ticker}{suffix}' for ticker in series.to_list()]

    ticker_list = [t for t in ticker_list if isinstance(t, str)]    # exclude nan
    return ticker_list


def get_ticker_list(country_code: str) -> list:
    ticker_list_path = save_root / f'{fn_ticker_list}.json'

    if ticker_list_path.is_file():
        print("Loading the ticker list...")
        with ticker_list_path.open('rt') as fp:
            ticker_list = json.load(fp)
    else:
        print("Download the ticker list...")
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
                market_cap_in_usd = usd_converter.convert(market_cap, currency, 'USD')    # TWD is not included.

            if market_cap_in_usd >= args.min_market_cap:
                result[k] = v
                result[k]['marketCap'] = market_cap

    return result


def remove_sector(input_dict: dict) -> dict:
    print(f"\nThe company sector in {excluded_sectors} will be excluded.")
    return {k: v for k, v in input_dict.items() if v['sector'] not in excluded_sectors}


def remove_country(input_dict: dict) -> dict:
    accepted_country = country_dict[args.country.upper()]
    print(f"\nThe company country not {accepted_country} will be excluded.")
    return {k: v for k, v in input_dict.items() if v['country'] == accepted_country}


def process_args():
    parser = argparse.ArgumentParser(description='Process refresh options')
    parser.add_argument('--country', '-c', choices=['US', 'TW'], type=str, default='US', dest='country',
                        help='Get stocks from which country')
    parser.add_argument('--min-market-cap', '-m', type=int, default=1e9, dest='min_market_cap',
                        help='Minimal market cap in USD')
    return parser.parse_args()


def main():
    # Retrieve or load ticker list
    ticker_list = get_ticker_list(args.country)
    financial_dict = {}

    # Load old financial data then update it
    for metric, keys, is_forced in zip(metric_list, keys_list, force_renew_list):
        financial_dict = get_financial(ticker_list, metric, keys, is_forced)

    # Exclude inappropriate tickers
    for f in filter_list:
        financial_dict = f(financial_dict)

    # Update database and rank stocks
    update_db(financial_dict, save_root / f"{fn_stock_rank}.db")
    rank_stocks(save_root / f"{fn_stock_rank}.db", save_root / f"{fn_stock_rank}.csv")


if __name__ == '__main__':
    args = process_args()
    save_root = Path(args.country.upper())
    save_root.mkdir(0o755, parents=True, exist_ok=True)

    country_dict = {'US': 'United States', 'TW': 'Taiwan'}
    fn_ticker_list = 'ticker_list'
    fn_financial = 'financial'
    fn_stock_rank = 'stock_rank'

    # https://yahooquery.dpguthrie.com/guide/ticker/modules/#summary_profile
    profile_keys = ["sector", "country"]

    # https://yahooquery.dpguthrie.com/guide/ticker/modules/#price
    price_keys = ["longName", "currency", "marketCap", "regularMarketPrice", "regularMarketTime"]

    # https://yahooquery.dpguthrie.com/guide/ticker/modules/#key_stats
    stat_keys = ["bookValue"]

    # https://yahooquery.dpguthrie.com/guide/ticker/financials/#get_financial_data
    financial_keys = ["asOfDate", "EBIT", "TotalAssets", "TotalDebt", "LongTermDebtAndCapitalLeaseObligation",
                      "CurrentAssets", "CurrentLiabilities", "NetPPE", "CashCashEquivalentsAndShortTermInvestments",
                      "MinorityInterest", "PreferredStock"]

    keys_list = [profile_keys, price_keys, stat_keys, financial_keys]
    all_keys = list(chain.from_iterable(keys_list))

    metric_list = ["profile", "price", "stat", "financial"]
    force_renew_list = [False, True, True, True]
    excluded_sectors = ["Financial Services", "Utilities", "Real Estate"]
    filter_list = [remove_outdated, remove_sector, remove_small_marketcap, remove_country]

    start = time.time()
    main()
    end = time.time()

    print(f"Execution time: {end - start:.1f} seconds")
