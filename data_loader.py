"""
数据加载模块
功能：获取股票列表、历史日线数据、数据清洗、过滤不合格股票
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import config


class DataLoader:
    """数据加载器"""

    def __init__(self):
        """初始化 Tushare 接口"""
        if not config.TUSHARE_TOKEN:
            raise ValueError("请在 config.py 中设置 TUSHARE_TOKEN")

        ts.set_token(config.TUSHARE_TOKEN)
        self.pro = ts.pro_api()

    def get_trade_cal(self, start_date=None, end_date=None):
        """
        获取交易日历

        Args:
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD

        Returns:
            DataFrame: 交易日历数据
        """
        if start_date is None:
            start_date = config.START_DATE
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        trade_cal = self.pro.trade_cal(
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open='1'
        )
        return trade_cal

    def get_latest_trade_date(self):
        """
        获取最近一个交易日

        Returns:
            str: 交易日期，格式 YYYYMMDD
        """
        today = datetime.now()
        for i in range(10):  # 最多往前查10天
            date_str = (today - timedelta(days=i)).strftime("%Y%m%d")
            trade_cal = self.pro.trade_cal(
                exchange='SSE',
                start_date=date_str,
                end_date=date_str,
                is_open='1'
            )
            if not trade_cal.empty:
                return date_str
        raise ValueError("无法获取最近交易日")

    def get_stock_list(self, trade_date=None):
        """
        获取股票列表，过滤 ST、退市、停牌、次新股

        Args:
            trade_date: 交易日期，格式 YYYYMMDD，默认为最近一个交易日

        Returns:
            DataFrame: 符合条件的股票列表
        """
        if trade_date is None:
            trade_date = self.get_latest_trade_date()

        print(f"获取 {trade_date} 的股票列表...")

        # 获取所有A股列表
        stock_basic = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )

        # 过滤条件
        valid_stocks = stock_basic.copy()

        # 1. 剔除 ST 股票
        if config.EXCLUDE_ST:
            valid_stocks = valid_stocks[~valid_stocks['name'].str.contains('ST|st|退', na=False)]

        # 2. 剔除次新股（上市不足N个交易日）
        if config.EXCLUDE_NEW_STOCK_DAYS > 0:
            trade_cal = self.get_trade_cal(
                start_date=config.START_DATE,
                end_date=trade_date
            )
            cutoff_date = trade_cal.iloc[-config.EXCLUDE_NEW_STOCK_DAYS]['cal_date']
            valid_stocks = valid_stocks[valid_stocks['list_date'] <= cutoff_date]

        # 3. 剔除停牌股票（通过日线数据检查）
        if config.EXCLUDE_SUSPENDED:
            valid_stocks = self._filter_suspended_stocks(valid_stocks, trade_date)

        print(f"筛选后股票数量: {len(valid_stocks)}")
        return valid_stocks

    def _filter_suspended_stocks(self, stock_list, trade_date):
        """
        过滤停牌股票

        Args:
            stock_list: 股票列表
            trade_date: 交易日期

        Returns:
            DataFrame: 未停牌的股票列表
        """
        valid_codes = []

        for idx, row in stock_list.iterrows():
            ts_code = row['ts_code']
            try:
                # 获取最近一天的数据
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=trade_date,
                    end_date=trade_date
                )
                if not df.empty:
                    valid_codes.append(ts_code)
                time.sleep(0.05)  # 限制请求频率
            except Exception as e:
                print(f"检查停牌失败 {ts_code}: {e}")
                continue

        return stock_list[stock_list['ts_code'].isin(valid_codes)]

    def get_stock_daily(self, ts_code, start_date=None, end_date=None):
        """
        获取单只股票日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 日线数据
        """
        if start_date is None:
            start_date = config.START_DATE
        if end_date is None:
            end_date = self.get_latest_trade_date()

        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                return None

            # 按日期排序
            df = df.sort_values('trade_date').reset_index(drop=True)

            # 数据清洗
            df = self._clean_daily_data(df)

            return df
        except Exception as e:
            print(f"获取 {ts_code} 数据失败: {e}")
            return None

    def _clean_daily_data(self, df):
        """
        清洗日线数据

        Args:
            df: 原始日线数据

        Returns:
            DataFrame: 清洗后的数据
        """
        # 删除缺失值
        df = df.dropna()

        # 确保价格合理
        df = df[
            (df['close'] > 0) &
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['vol'] >= 0)
        ]

        # 确保价格逻辑正确
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['close']) &
            (df['high'] >= df['open']) &
            (df['low'] <= df['close']) &
            (df['low'] <= df['open'])
        ]

        return df

    def get_all_stocks_daily(self, stock_list, start_date=None, end_date=None):
        """
        批量获取股票日线数据

        Args:
            stock_list: 股票列表 DataFrame
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            dict: {ts_code: DataFrame} 格式的字典
        """
        all_data = {}
        total = len(stock_list)

        print(f"开始获取 {total} 只股票的日线数据...")

        for idx, row in stock_list.iterrows():
            ts_code = row['ts_code']
            name = row['name']

            df = self.get_stock_daily(ts_code, start_date, end_date)

            if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
                all_data[ts_code] = df
                print(f"[{len(all_data)}/{total}] {ts_code} {name}: {len(df)} 条数据")
            else:
                print(f"[跳过] {ts_code} {name}: 数据不足")

            # 控制请求频率
            time.sleep(0.2)

        print(f"成功获取 {len(all_data)} 只股票的数据")
        return all_data

    def filter_by_liquidity(self, stock_data, trade_date):
        """
        根据流动性筛选股票

        Args:
            stock_data: 股票数据字典
            trade_date: 交易日期

        Returns:
            dict: 筛选后的股票数据
        """
        filtered_data = {}

        for ts_code, df in stock_data.items():
            # 获取指定日期的数据
            day_data = df[df['trade_date'] == trade_date]

            if day_data.empty:
                continue

            row = day_data.iloc[0]

            # 流动性筛选
            if row['vol'] < config.MIN_VOLUME:
                continue
            if row['amount'] < config.MIN_AMOUNT:
                continue

            
            # 涨跌停筛选
            if config.EXCLUDE_LIMIT_UP and row['pct_chg'] > config.LIMIT_THRESHOLD * 100:
                continue
            if config.EXCLUDE_LIMIT_DOWN and row['pct_chg'] < -config.LIMIT_THRESHOLD * 100:
                continue

            filtered_data[ts_code] = df

        print(f"流动性筛选后: {len(filtered_data)} 只股票")
        return filtered_data

    def calculate_volatility(self, df, period=None):
        """
        计算波动率

        Args:
            df: 日线数据
            period: 计算周期

        Returns:
            float: 波动率
        """
        if period is None:
            period = config.MAX_VOLATILITY_DAYS

        if len(df) < period:
            return np.nan

        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(period).std()

        return volatility

    def filter_by_volatility(self, stock_data):
        """
        根据波动率筛选股票

        Args:
            stock_data: 股票数据字典

        Returns:
            dict: 筛选后的股票数据
        """
        filtered_data = {}

        for ts_code, df in stock_data.items():
            volatility = self.calculate_volatility(df)

            if np.isnan(volatility) or volatility > config.MAX_VOLATILITY_THRESHOLD:
                continue

            filtered_data[ts_code] = df

        print(f"波动率筛选后: {len(filtered_data)} 只股票")
        return filtered_data


def main():
    """测试数据加载功能"""
    loader = DataLoader()

    # 获取最近交易日
    trade_date = loader.get_latest_trade_date()
    print(f"最近交易日: {trade_date}")

    # 获取股票列表
    stock_list = loader.get_stock_list(trade_date)
    print(f"符合条件的股票: {len(stock_list)} 只")

    # 测试获取单只股票数据
    if not stock_list.empty:
        test_code = stock_list.iloc[0]['ts_code']
        test_name = stock_list.iloc[0]['name']
        df = loader.get_stock_daily(test_code)
        if df is not None:
            print(f"\n{test_code} {test_name} 数据样例:")
            print(df.tail())


if __name__ == "__main__":
    main()