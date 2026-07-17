"""
鏁版嵁鍔犺浇妯″潡
鍔熻兘锛氳幏鍙栬偂绁ㄥ垪琛ㄣ€佸巻鍙叉棩绾挎暟鎹€佹暟鎹竻娲椼€佽繃婊や笉鍚堟牸鑲＄エ
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import config


class DataLoader:
    """Data loader."""

    def __init__(self):
        """Initialize Tushare API."""
        if not config.TUSHARE_TOKEN:
            raise ValueError("璇峰湪 config.py 涓缃?TUSHARE_TOKEN")

        ts.set_token(config.TUSHARE_TOKEN)
        self.pro = ts.pro_api(timeout=30)

    def get_trade_cal(self, start_date=None, end_date=None):
        """
        鑾峰彇浜ゆ槗鏃ュ巻

        Args:
            start_date: 寮€濮嬫棩鏈燂紝鏍煎紡 YYYYMMDD
            end_date: 缁撴潫鏃ユ湡锛屾牸寮?YYYYMMDD

        Returns:
            DataFrame: 浜ゆ槗鏃ュ巻鏁版嵁
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

    def get_latest_trade_date(self, as_of_date=None):
        """
        鑾峰彇鏈€杩戜竴涓氦鏄撴棩

        Returns:
            str: 浜ゆ槗鏃ユ湡锛屾牸寮?YYYYMMDD
        """
        end_date = (
            str(as_of_date)
            if as_of_date is not None
            else datetime.now().strftime("%Y%m%d")
        )
        lookback_days = int(
            getattr(config, 'LATEST_TRADE_DATE_LOOKBACK_DAYS', 15) or 15
        )
        start_date = (
            datetime.strptime(end_date, "%Y%m%d")
            - timedelta(days=lookback_days)
        ).strftime("%Y%m%d")
        trade_cal = self.pro.trade_cal(
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open='1'
        )
        if trade_cal.empty or 'cal_date' not in trade_cal.columns:
            raise ValueError("无法获取最近交易日历")

        min_rows = int(getattr(config, 'LATEST_DAILY_MIN_ROWS', 100) or 100)
        candidates = sorted(
            trade_cal['cal_date'].astype(str).unique(), reverse=True
        )
        errors = []
        for date_str in candidates:
            cache_file = os.path.join(
                config.DAILY_CACHE_DIR, f"{date_str}.csv"
            )
            if os.path.exists(cache_file):
                try:
                    cached = pd.read_csv(
                        cache_file, usecols=['ts_code'], nrows=min_rows
                    )
                    if len(cached) >= min_rows:
                        return date_str
                except (OSError, ValueError, pd.errors.ParserError):
                    pass

            try:
                daily = self.pro.daily(
                    trade_date=date_str,
                    fields='ts_code,trade_date,close'
                )
            except Exception as exc:
                errors.append(f"{date_str}: {exc}")
                continue
            if daily is not None and len(daily) >= min_rows:
                if date_str != candidates[0]:
                    print(
                        f"{candidates[0]} 日线尚未发布，"
                        f"自动使用最近完整交易日 {date_str}"
                    )
                return date_str

        detail = f"；接口错误: {errors[-1]}" if errors else ""
        raise ValueError(f"最近 {lookback_days} 天无完整日线数据{detail}")

    def get_stock_list(
        self, trade_date=None, include_sector=True,
        include_market_cap=True, historical=False
    ):
        """获取指定时点股票池；训练模式同时纳入历史退市股票。"""
        if trade_date is None:
            trade_date = self.get_latest_trade_date()
        trade_date = str(trade_date)

        statuses = ('L', 'D', 'P') if historical else ('L', 'D', 'P')
        frames = []
        fields = 'ts_code,symbol,name,area,industry,list_status,list_date,delist_date'
        for status in statuses:
            df_status = self.pro.stock_basic(
                exchange='', list_status=status, fields=fields
            )
            if df_status is not None and not df_status.empty:
                frames.append(df_status)
        if not frames:
            raise ValueError("无法获取股票基础列表")

        stock_basic = pd.concat(frames, ignore_index=True).drop_duplicates('ts_code')
        stock_basic['list_date'] = stock_basic['list_date'].fillna('').astype(str)
        stock_basic['delist_date'] = stock_basic['delist_date'].fillna('').astype(str)

        if historical:
            start_date = str(config.START_DATE)
            active_mask = (
                (stock_basic['list_date'] <= trade_date)
                & ((stock_basic['delist_date'] == '') | (stock_basic['delist_date'] >= start_date))
            )
        else:
            active_mask = (
                (stock_basic['list_date'] <= trade_date)
                & ((stock_basic['delist_date'] == '') | (stock_basic['delist_date'] >= trade_date))
            )
        valid_stocks = stock_basic.loc[active_mask].copy()

        valid_stocks = valid_stocks[~valid_stocks['ts_code'].str.endswith('.BJ')]
        exclude_prefixes = tuple(getattr(config, 'EXCLUDE_CODE_PREFIXES', ()) or ())
        if exclude_prefixes:
            before_count = len(valid_stocks)
            symbols = valid_stocks['ts_code'].str.split('.').str[0]
            valid_stocks = valid_stocks[~symbols.str.startswith(exclude_prefixes)]
            print(f"排除不可交易代码前缀 {exclude_prefixes}: {before_count} -> {len(valid_stocks)}")

        # 历史训练在逐日数据上判断ST，不能用今天的名称删除过去样本。
        if config.EXCLUDE_ST and not historical:
            invalid_name_pattern = r'\*?ST|退|退市'
            valid_stocks = valid_stocks[
                ~valid_stocks['name'].str.contains(
                    invalid_name_pattern, case=False, na=False, regex=True
                )
            ]

        if config.EXCLUDE_NEW_STOCK_DAYS > 0 and not historical:
            trade_cal = self.get_trade_cal(start_date=config.START_DATE, end_date=trade_date)
            if len(trade_cal) >= config.EXCLUDE_NEW_STOCK_DAYS:
                cutoff_date = trade_cal.iloc[-config.EXCLUDE_NEW_STOCK_DAYS]['cal_date']
                valid_stocks = valid_stocks[valid_stocks['list_date'] <= cutoff_date]

        if include_sector and 'industry' in valid_stocks.columns:
            industry = (
                valid_stocks['industry']
                .fillna('')
                .astype(str)
                .str.strip()
                .replace('', config.DEFAULT_SECTOR)
            )
            mapped_sector = industry.map(config.INDUSTRY_MAP)
            valid_stocks['sector'] = mapped_sector.where(
                mapped_sector.notna(), industry
            )
        else:
            valid_stocks['sector'] = config.DEFAULT_SECTOR

        valid_stocks['market_cap'] = np.nan
        if include_market_cap:
            try:
                daily_basic = self.pro.daily_basic(
                    trade_date=trade_date, fields='ts_code,circ_mv,total_mv'
                )
                if daily_basic is not None and not daily_basic.empty:
                    valid_stocks = valid_stocks.drop(columns=['market_cap']).merge(
                        daily_basic[['ts_code', 'circ_mv', 'total_mv']],
                        on='ts_code', how='left'
                    )
                    valid_stocks['market_cap'] = valid_stocks['circ_mv']
            except Exception as e:
                print(f"获取 {trade_date} 市值数据失败: {e}")

        if config.EXCLUDE_SUSPENDED and not historical:
            valid_stocks = self._filter_suspended_stocks(valid_stocks, trade_date)

        stock_pool_limit = getattr(config, 'STOCK_POOL_LIMIT', 0)
        if not historical and stock_pool_limit and valid_stocks['market_cap'].notna().any():
            valid_stocks = (
                valid_stocks.sort_values('market_cap', ascending=False)
                .head(stock_pool_limit)
            )
            print(f"按 {trade_date} 流通市值保留前 {stock_pool_limit} 只股票")

        valid_stocks = valid_stocks.reset_index(drop=True)
        mode = "历史训练" if historical else "时点预测"
        print(f"{mode}股票池数量: {len(valid_stocks)}")
        return valid_stocks

    def _get_name_changes(self):
        """获取并缓存历史名称区间，用于逐日识别ST状态。"""
        cache_file = os.path.join(config.CACHE_DIR, 'namechange.csv')
        if os.path.exists(cache_file):
            return pd.read_csv(
                cache_file,
                dtype={'ts_code': str, 'start_date': str, 'end_date': str}
            )

        try:
            frames = []
            current_year = datetime.now().year
            for year in range(1990, current_year + 1):
                frame = self.pro.namechange(
                    start_date=f'{year}0101', end_date=f'{year}1231'
                )
                if frame is not None and not frame.empty:
                    frames.append(frame)
                time.sleep(0.05)
            if not frames:
                return pd.DataFrame()
            name_changes = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(['ts_code', 'name', 'start_date', 'end_date'])
            )
            name_changes.to_csv(cache_file, index=False, encoding='utf-8-sig')
            return name_changes
        except Exception as e:
            print(f"警告: 历史名称获取失败，训练将保留未知ST样本: {e}")
            return pd.DataFrame()

    @staticmethod
    def _apply_stock_metadata(df, meta, name_changes=None):
        """为逐日行情附加股票池元数据和历史ST标记。"""
        df = df.copy()

        def _text(value, default=''):
            return default if value is None or pd.isna(value) or str(value) == 'nan' else str(value)

        df['industry'] = _text(meta.get('industry'), config.DEFAULT_SECTOR)
        df['sector'] = _text(meta.get('sector'), config.DEFAULT_SECTOR)
        df['list_date'] = _text(meta.get('list_date'))
        df['delist_date'] = _text(meta.get('delist_date'))
        df['historical_name'] = _text(meta.get('name'))
        df['is_st'] = False

        if name_changes is not None and not name_changes.empty:
            changes = name_changes[name_changes['ts_code'] == meta['ts_code']]
            for _, change in changes.iterrows():
                start = str(change.get('start_date') or '')
                end = str(change.get('end_date') or '99991231')
                if not start or start == 'nan':
                    continue
                if not end or end == 'nan' or end == 'None':
                    end = '99991231'
                mask = df['trade_date'].astype(str).between(start, end)
                name = str(change.get('name') or '')
                df.loc[mask, 'historical_name'] = name
                if pd.Series([name]).str.contains(r'\*?ST|退|退市', case=False, regex=True).iloc[0]:
                    df.loc[mask, 'is_st'] = True

        if df['list_date'].iloc[0]:
            trade_dt = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
            list_dt = pd.to_datetime(df['list_date'].iloc[0], format='%Y%m%d', errors='coerce')
            df['listing_age_days'] = (trade_dt - list_dt).dt.days
        else:
            df['listing_age_days'] = np.nan

        if 'circ_mv' in df.columns:
            df['market_cap'] = pd.to_numeric(df['circ_mv'], errors='coerce')
        else:
            # 截止日市值只能用于最后一个时点，禁止回填到整段历史。
            df['market_cap'] = np.nan
            latest_market_cap = pd.to_numeric(
                pd.Series([meta.get('market_cap', np.nan)]), errors='coerce'
            ).iloc[0]
            if pd.notna(latest_market_cap) and not df.empty:
                latest_idx = df['trade_date'].astype(str).idxmax()
                df.loc[latest_idx, 'market_cap'] = latest_market_cap
        return df

    @staticmethod
    def _apply_adjusted_prices(df):
        """保留原始成交价，并将特征价格替换为截止日锚定的前复权价格。"""
        if not getattr(config, 'USE_ADJUSTED_PRICES', False):
            return df
        if 'adj_factor' not in df.columns or df['adj_factor'].notna().sum() == 0:
            if getattr(config, 'REQUIRE_ADJ_FACTOR', True):
                raise ValueError("缺少复权因子")
            return df

        df = df.sort_values('trade_date').reset_index(drop=True).copy()
        factor = pd.to_numeric(df['adj_factor'], errors='coerce').ffill()
        if factor.isna().any():
            raise ValueError("复权因子存在无法用历史值填充的缺口")
        anchor = factor.iloc[-1]
        if not np.isfinite(anchor) or anchor <= 0:
            raise ValueError("无效复权因子")
        ratio = factor / anchor

        for col in ('open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg'):
            if col in df.columns:
                df[f'raw_{col}'] = df[col]
        for col in ('open', 'high', 'low', 'close'):
            df[col] = pd.to_numeric(df[col], errors='coerce') * ratio

        shifted_close = df['close'].shift(1)
        first_pre_close = pd.to_numeric(df['raw_pre_close'], errors='coerce') * ratio
        df['pre_close'] = shifted_close.fillna(first_pre_close)
        df['change'] = df['close'] - df['pre_close']
        df['pct_chg'] = df['change'] / df['pre_close'].replace(0, np.nan) * 100
        return df

    def _filter_suspended_stocks(self, stock_list, trade_date):
        """
        杩囨护鍋滅墝鑲＄エ

        Args:
            stock_list: 鑲＄エ鍒楄〃
            trade_date: 浜ゆ槗鏃ユ湡

        Returns:
            DataFrame: 鏈仠鐗岀殑鑲＄エ鍒楄〃
        """
        valid_codes = []

        for idx, row in stock_list.iterrows():
            ts_code = row['ts_code']
            try:
                # 鑾峰彇鏈€杩戜竴澶╃殑鏁版嵁
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=trade_date,
                    end_date=trade_date
                )
                if not df.empty:
                    valid_codes.append(ts_code)
                time.sleep(0.2)  # 闄愬埗璇锋眰棰戠巼锛岄伩鍏嶈Е鍙?Tushare 闄愭祦
            except Exception as e:
                print(f"妫€鏌ュ仠鐗屽け璐?{ts_code}: {e}")
                continue

        return stock_list[stock_list['ts_code'].isin(valid_codes)]

    def get_stock_daily(self, ts_code, start_date=None, end_date=None):
        """
        鑾峰彇鍗曞彧鑲＄エ鏃ョ嚎鏁版嵁

        Args:
            ts_code: 鑲＄エ浠ｇ爜
            start_date: 寮€濮嬫棩鏈?            end_date: 缁撴潫鏃ユ湡

        Returns:
            DataFrame: 鏃ョ嚎鏁版嵁
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

            df = df.sort_values('trade_date').reset_index(drop=True)

            # 鏁版嵁娓呮礂
            df = self._clean_daily_data(df)

            if getattr(config, 'USE_ADJUSTED_PRICES', False):
                adj = self.pro.adj_factor(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                if adj is not None and not adj.empty:
                    adj['trade_date'] = adj['trade_date'].astype(str)
                    df['trade_date'] = df['trade_date'].astype(str)
                    df = df.merge(
                        adj[['trade_date', 'adj_factor']],
                        on='trade_date', how='left'
                    )
                df = self._apply_adjusted_prices(df)

            return df
        except Exception as e:
            print(f"鑾峰彇 {ts_code} 鏁版嵁澶辫触: {e}")
            return None

    def _clean_daily_data(self, df):
        """
        娓呮礂鏃ョ嚎鏁版嵁

        Args:
            df: 鍘熷鏃ョ嚎鏁版嵁

        Returns:
            DataFrame: 娓呮礂鍚庣殑鏁版嵁
        """
        required = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        df = df.dropna(subset=[col for col in required if col in df.columns])

        # 纭繚浠锋牸鍚堢悊
        df = df[
            (df['close'] > 0) &
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['vol'] >= 0)
        ]

        # 纭繚浠锋牸閫昏緫姝ｇ‘
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['close']) &
            (df['high'] >= df['open']) &
            (df['low'] <= df['close']) &
            (df['low'] <= df['open'])
        ]

        return df

    def get_all_stocks_daily(self, stock_list, start_date=None, end_date=None, include_basic=True):
        """
        鎵归噺鑾峰彇鑲＄エ鏃ョ嚎鏁版嵁锛堜紭鍏堜娇鐢ㄥ叏甯傚満鎺ュ彛锛屽噺灏慉PI璋冪敤娆℃暟锛?        鍙€夊悎骞?daily_basic / moneyflow / top_list / limit_list 绛夐珮闃舵暟鎹?
        Args:
            stock_list: 鑲＄エ鍒楄〃 DataFrame
            start_date: 寮€濮嬫棩鏈?            end_date: 缁撴潫鏃ユ湡
            include_basic: 鏄惁鍚堝苟 daily_basic 绛夐珮闃舵寚鏍?
        Returns:
            dict: {ts_code: DataFrame} 鏍煎紡鐨勫瓧鍏?        """
        if start_date is None:
            start_date = config.START_DATE
        if end_date is None:
            end_date = self.get_latest_trade_date()

        total = len(stock_list)
        print(f"寮€濮嬫壒閲忚幏鍙?{total} 鍙偂绁ㄧ殑鏃ョ嚎鏁版嵁 ({start_date} ~ {end_date})...")

        stock_meta = {
            row['ts_code']: row.to_dict()
            for _, row in stock_list.iterrows()
        }
        name_changes = None
        if getattr(config, 'USE_HISTORICAL_ST_FILTER', False):
            name_changes = self._get_name_changes()

        if total <= getattr(config, 'FETCH_BY_STOCK_THRESHOLD', 0):
            print("灏忔牱鏈ā寮忥細鎸夎偂绁ㄤ覆琛岃幏鍙栨棩绾匡紝璺宠繃鍏ㄥ競鍦洪€愭棩鎷夊彇")
            return self._get_all_stocks_daily_serial(stock_list, start_date, end_date)

        # 鑾峰彇楂橀樁杈呭姪鏁版嵁
        df_basic_all = None
        df_moneyflow_all = None
        df_top_list_all = None
        df_limit_list_all = None

        if include_basic:
            try:
                df_basic_all = self._get_daily_basic_by_date(start_date, end_date)
            except Exception as e:
                print(f"姣忔棩鎸囨爣鑾峰彇澶辫触: {e}")
            if getattr(config, 'INCLUDE_MONEYFLOW', True):
                try:
                    df_moneyflow_all = self._get_moneyflow_by_date(start_date, end_date)
                except Exception as e:
                    print(f"璧勯噾娴佸悜鑾峰彇澶辫触: {e}")
            if getattr(config, 'INCLUDE_TOP_LIST', True):
                try:
                    df_top_list_all = self._get_top_list_by_date(start_date, end_date)
                except Exception as e:
                    print(f"榫欒檸姒滆幏鍙栧け璐? {e}")
            if getattr(config, 'INCLUDE_LIMIT_LIST', True):
                try:
                    df_limit_list_all = self._get_limit_list_by_date(start_date, end_date)
                except Exception as e:
                    print(f"娑ㄨ穼鍋滅粺璁¤幏鍙栧け璐? {e}")

        # 绛栫暐1锛氭寜浜ゆ槗鏃ュ垎鐗囪幏鍙栧叏甯傚満鏃ョ嚎鏁版嵁
        try:
            df_all = self._get_all_stocks_daily_by_date(start_date, end_date)

            if df_all is not None and not df_all.empty:
                all_data = {}
                target_codes = set(stock_list['ts_code'])

                for ts_code, group_df in df_all.groupby('ts_code'):
                    if ts_code not in target_codes:
                        continue

                    df = group_df.sort_values('trade_date').reset_index(drop=True).copy()

                    # 鍚堝苟 daily_basic 鏁版嵁
                    if df_basic_all is not None and not df_basic_all.empty:
                        basic_df = df_basic_all[df_basic_all['ts_code'] == ts_code].copy()
                        if not basic_df.empty:
                            merge_cols = [c for c in basic_df.columns if c not in ['ts_code', 'trade_date']]
                            df = df.merge(basic_df[['trade_date'] + merge_cols], on='trade_date', how='left')

                    # 鍚堝苟璧勯噾娴佸悜鏁版嵁
                    if df_moneyflow_all is not None and not df_moneyflow_all.empty:
                        mf_df = df_moneyflow_all[df_moneyflow_all['ts_code'] == ts_code].copy()
                        if not mf_df.empty:
                            merge_cols = [c for c in mf_df.columns if c not in ['ts_code', 'trade_date']]
                            df = df.merge(mf_df[['trade_date'] + merge_cols], on='trade_date', how='left')

                    if df_top_list_all is not None and not df_top_list_all.empty:
                        top_df = df_top_list_all[df_top_list_all['ts_code'] == ts_code].copy()
                        if not top_df.empty:
                            merge_cols = [
                                c for c in top_df.columns
                                if c not in {
                                    'ts_code', 'trade_date', 'name', 'close',
                                    'amount', 'pct_change', 'pct_chg'
                                }
                            ]
                            df = df.merge(top_df[['trade_date'] + merge_cols], on='trade_date', how='left')

                    if df_limit_list_all is not None and not df_limit_list_all.empty:
                        limit_df = df_limit_list_all[df_limit_list_all['ts_code'] == ts_code].copy()
                        if not limit_df.empty:
                            merge_cols = [
                                c for c in limit_df.columns
                                if c not in {
                                    'ts_code', 'trade_date', 'name', 'close',
                                    'amount', 'pct_change', 'pct_chg'
                                }
                            ]
                            df = df.merge(limit_df[['trade_date'] + merge_cols], on='trade_date', how='left')

                    sparse_numeric = [
                        'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
                        'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
                        'net_mf_vol', 'net_mf_amount', 'net_amount', 'l_amount',
                        'net_rate', 'fd_amount', 'fc_ratio', 'open_times', 'strth'
                    ]
                    for col in sparse_numeric:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    for col in ['turnover_rate', 'turnover_rate_f', 'volume_ratio',
                                'pe', 'pe_ttm', 'pb', 'ps', 'total_mv', 'circ_mv']:
                        if col in df.columns:
                            values = pd.to_numeric(df[col], errors='coerce')
                            df[col] = values.ffill()
                    for col in ['limit', 'reason']:
                        if col in df.columns:
                            df[col] = df[col].fillna('')

                    df = df.sort_values('trade_date').reset_index(drop=True)
                    if df.duplicated('trade_date').any():
                        raise ValueError(f"{ts_code} 合并后存在重复交易日")
                    df = self._clean_daily_data(df)
                    df = self._apply_adjusted_prices(df)
                    meta = stock_meta.get(ts_code, {'ts_code': ts_code})
                    df = self._apply_stock_metadata(df, meta, name_changes)

                    if len(df) >= config.MIN_HISTORY_DAYS:
                        all_data[ts_code] = df

                print(f"鎵归噺鎺ュ彛鎴愬姛鑾峰彇 {len(all_data)} 鍙偂绁ㄧ殑鏁版嵁")
                if len(all_data) > 0:
                    return all_data
                raise ValueError("批量接口没有生成可用股票数据")

        except Exception as e:
            raise RuntimeError(f"全市场批量数据构建失败: {e}") from e

    def _get_adj_factor_for_date(self, trade_date):
        """获取单日全市场复权因子并缓存。"""
        cache_dir = os.path.join(config.CACHE_DIR, 'adj_factor')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{trade_date}.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(
                cache_file,
                dtype={'ts_code': str, 'trade_date': str}
            )

        df_adj = self.pro.adj_factor(trade_date=trade_date)
        if df_adj is not None and not df_adj.empty:
            df_adj.to_csv(cache_file, index=False, encoding='utf-8-sig')
            time.sleep(0.1)
        return df_adj

    def _get_all_stocks_daily_by_date(self, start_date, end_date):
        """
        鎸変氦鏄撴棩鍒嗙墖鑾峰彇鍏ㄥ競鍦烘棩绾挎暟鎹紙閬垮厤鍗曡姹傛暟鎹噺杩囧ぇ瀵艰嚧瓒呮椂鎴栧垎椤甸仐婕忥級

        Returns:
            DataFrame: 鍚堝苟鍚庣殑鍏ㄥ競鍦烘棩绾挎暟鎹?        """
        trade_cal = self.get_trade_cal(start_date=start_date, end_date=end_date)
        if trade_cal is None or trade_cal.empty:
            return pd.DataFrame()

        all_dfs = []
        failed_dates = []
        dates = trade_cal['cal_date'].tolist()
        print(f"鎸変氦鏄撴棩鍒嗙墖鑾峰彇鏃ョ嚎锛屽叡 {len(dates)} 涓氦鏄撴棩...")

        for i, trade_date in enumerate(dates):
            try:
                cache_file = os.path.join(config.DAILY_CACHE_DIR, f"{trade_date}.csv")
                cache_exists = os.path.exists(cache_file)
                if cache_exists:
                    df_day = pd.read_csv(cache_file, dtype={'trade_date': str})
                else:
                    df_day = self.pro.daily(trade_date=trade_date)
                    if df_day is not None and not df_day.empty:
                        df_day.to_csv(cache_file, index=False, encoding='utf-8-sig')

                if df_day is not None and not df_day.empty:
                    df_day['trade_date'] = df_day['trade_date'].astype(str)
                    required_cols = {
                        'ts_code', 'trade_date', 'open', 'high', 'low',
                        'close', 'vol', 'amount'
                    }
                    missing_cols = required_cols - set(df_day.columns)
                    if missing_cols:
                        raise ValueError(f"缓存缺少字段: {sorted(missing_cols)}")
                    if df_day.duplicated(['ts_code', 'trade_date']).any():
                        raise ValueError("缓存存在重复股票日线")
                    if getattr(config, 'USE_ADJUSTED_PRICES', False):
                        df_adj = self._get_adj_factor_for_date(trade_date)
                        if df_adj is None or df_adj.empty:
                            if getattr(config, 'REQUIRE_ADJ_FACTOR', True):
                                raise ValueError(f"{trade_date} 缺少复权因子")
                        else:
                            df_adj['trade_date'] = df_adj['trade_date'].astype(str)
                            df_day = df_day.merge(
                                df_adj[['ts_code', 'trade_date', 'adj_factor']],
                                on=['ts_code', 'trade_date'], how='left'
                            )
                    all_dfs.append(df_day)
                else:
                    failed_dates.append(str(trade_date))
                if (i + 1) % 10 == 0 or (i + 1) == len(dates):
                    print(f"  daily fetched {i+1}/{len(dates)} dates, rows={sum(len(d) for d in all_dfs)}")
                if not cache_exists:
                    time.sleep(0.1)
            except Exception as e:
                print(f"  鑾峰彇 {trade_date} 鏃ョ嚎澶辫触: {e}")
                failed_dates.append(str(trade_date))

        if failed_dates:
            raise RuntimeError(
                f"日线数据缺失 {len(set(failed_dates))} 个交易日: "
                f"{sorted(set(failed_dates))[:10]}"
            )

        if not all_dfs:
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"daily data merged: {len(df_all)} rows")
        return df_all

    def _get_daily_basic_by_date(self, start_date, end_date):
        """
        鎸変氦鏄撴棩鍒嗙墖鑾峰彇鍏ㄥ競鍦烘瘡鏃ユ寚鏍囷紙daily_basic锛?        鍖呭惈锛歅E銆丳B銆佹崲鎵嬬巼銆侀噺姣旂瓑鍩烘湰闈?鎯呯华鎸囨爣

        Returns:
            DataFrame: 鍚堝苟鍚庣殑 daily_basic 鏁版嵁
        """
        trade_cal = self.get_trade_cal(start_date=start_date, end_date=end_date)
        if trade_cal is None or trade_cal.empty:
            return pd.DataFrame()

        all_dfs = []
        dates = trade_cal['cal_date'].tolist()
        print(f"鎸変氦鏄撴棩鍒嗙墖鑾峰彇姣忔棩鎸囨爣锛坉aily_basic锛夛紝鍏?{len(dates)} 涓氦鏄撴棩...")
        cache_dir = os.path.join(config.CACHE_DIR, "daily_basic")
        os.makedirs(cache_dir, exist_ok=True)

        for i, trade_date in enumerate(dates):
            try:
                cache_file = os.path.join(cache_dir, f"{trade_date}.csv")
                cache_exists = os.path.exists(cache_file)
                if cache_exists:
                    df_day = pd.read_csv(cache_file, dtype={'trade_date': str})
                else:
                    df_day = self.pro.daily_basic(trade_date=trade_date)
                    if df_day is not None and not df_day.empty:
                        df_day.to_csv(cache_file, index=False, encoding='utf-8-sig')
                if df_day is not None and not df_day.empty:
                    keep_cols = ['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f',
                                 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'total_mv', 'circ_mv']
                    available = [c for c in keep_cols if c in df_day.columns]
                    all_dfs.append(df_day[available])
                if (i + 1) % 10 == 0 or (i + 1) == len(dates):
                    print(f"  daily_basic fetched {i+1}/{len(dates)} dates, rows={sum(len(d) for d in all_dfs)}")
                if not cache_exists:
                    time.sleep(0.35)
            except Exception as e:
                print(f"  鑾峰彇 {trade_date} 姣忔棩鎸囨爣澶辫触: {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"daily_basic merged: {len(df_all)} rows")
        return df_all

    def _get_moneyflow_by_date(self, start_date, end_date):
        """
        鎸変氦鏄撴棩鍒嗙墖鑾峰彇鍏ㄥ競鍦鸿祫閲戞祦鍚戯紙moneyflow锛?        鍖呭惈锛氳秴澶у崟/澶у崟/涓崟/灏忓崟鍑€娴佸叆
        2000绉垎閫氬父鍙敤锛屼絾绉垎涓嶈冻鏃朵細鑷姩璺宠繃
        """
        trade_cal = self.get_trade_cal(start_date=start_date, end_date=end_date)
        if trade_cal is None or trade_cal.empty:
            return pd.DataFrame()

        all_dfs = []
        dates = trade_cal['cal_date'].tolist()
        print(f"鎸変氦鏄撴棩鍒嗙墖鑾峰彇璧勯噾娴佸悜锛坢oneyflow锛夛紝鍏?{len(dates)} 涓氦鏄撴棩...")
        cache_dir = os.path.join(config.CACHE_DIR, "moneyflow")
        os.makedirs(cache_dir, exist_ok=True)

        for i, trade_date in enumerate(dates):
            try:
                cache_file = os.path.join(cache_dir, f"{trade_date}.csv")
                cache_exists = os.path.exists(cache_file)
                if cache_exists:
                    df_day = pd.read_csv(cache_file, dtype={'trade_date': str})
                else:
                    df_day = self.pro.moneyflow(trade_date=trade_date)
                    if df_day is not None and not df_day.empty:
                        df_day.to_csv(cache_file, index=False, encoding='utf-8-sig')
                if df_day is not None and not df_day.empty:
                    keep_cols = ['ts_code', 'trade_date',
                                 'buy_sm_vol', 'sell_sm_vol',
                                 'buy_md_vol', 'sell_md_vol',
                                 'buy_lg_vol', 'sell_lg_vol',
                                 'buy_elg_vol', 'sell_elg_vol',
                                 'net_mf_vol', 'net_mf_amount']
                    available = [c for c in keep_cols if c in df_day.columns]
                    all_dfs.append(df_day[available])
                if (i + 1) % 20 == 0 or (i + 1) == len(dates):
                    print(f"  moneyflow fetched {i+1}/{len(dates)} dates, rows={sum(len(d) for d in all_dfs)}")
                if not cache_exists:
                    time.sleep(0.35)
            except Exception as e:
                err = str(e)
                if '绉垎' in err or '鏉冮檺' in err or 'Permission' in err:
                    print(f"  绉垎/鏉冮檺涓嶈冻锛岃烦杩囪祫閲戞祦鍚戣幏鍙? {err}")
                    return pd.DataFrame()
                print(f"  鑾峰彇 {trade_date} 璧勯噾娴佸悜澶辫触: {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"moneyflow merged: {len(df_all)} rows")
        return df_all

    def _get_top_list_by_date(self, start_date, end_date):
        """
        鎸変氦鏄撴棩鍒嗙墖鑾峰彇榫欒檸姒滄暟鎹紙top_list锛?        杩斿洖锛氫笂姒滆偂绁ㄣ€佸噣涔板叆棰濄€侀緳铏庢鎴愪氦棰濈瓑
        """
        trade_cal = self.get_trade_cal(start_date=start_date, end_date=end_date)
        if trade_cal is None or trade_cal.empty:
            return pd.DataFrame()

        all_dfs = []
        dates = trade_cal['cal_date'].tolist()
        print(f"鎸変氦鏄撴棩鍒嗙墖鑾峰彇榫欒檸姒滐紙top_list锛夛紝鍏?{len(dates)} 涓氦鏄撴棩...")
        cache_dir = os.path.join(config.CACHE_DIR, "top_list")
        os.makedirs(cache_dir, exist_ok=True)

        for i, trade_date in enumerate(dates):
            try:
                cache_file = os.path.join(cache_dir, f"{trade_date}.csv")
                cache_exists = os.path.exists(cache_file)
                if cache_exists:
                    df_day = pd.read_csv(cache_file, dtype={'trade_date': str})
                else:
                    df_day = self.pro.top_list(trade_date=trade_date)
                    if df_day is not None and not df_day.empty:
                        df_day.to_csv(cache_file, index=False, encoding='utf-8-sig')
                if df_day is not None and not df_day.empty:
                    keep_cols = ['ts_code', 'trade_date', 'name', 'close', 'pct_change',
                                 'amount', 'l_amount', 'net_amount', 'net_rate', 'reason']
                    available = [c for c in keep_cols if c in df_day.columns]
                    all_dfs.append(df_day[available])
                if (i + 1) % 30 == 0 or (i + 1) == len(dates):
                    print(f"  top_list fetched {i+1}/{len(dates)} dates, rows={sum(len(d) for d in all_dfs)}")
                if not cache_exists:
                    time.sleep(0.35)
            except Exception as e:
                print(f"  鑾峰彇 {trade_date} 榫欒檸姒滃け璐? {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True)
        aggregations = {}
        for col in ['l_amount', 'net_amount']:
            if col in df_all.columns:
                aggregations[col] = 'sum'
        if 'net_rate' in df_all.columns:
            aggregations['net_rate'] = 'mean'
        if 'reason' in df_all.columns:
            aggregations['reason'] = lambda values: '|'.join(
                sorted({str(value) for value in values if pd.notna(value)})
            )
        if aggregations:
            df_all = (
                df_all.groupby(['ts_code', 'trade_date'], as_index=False)
                .agg(aggregations)
            )
        print(f"top_list merged: {len(df_all)} rows")
        return df_all

    def _get_limit_list_by_date(self, start_date, end_date):
        """
        鎸変氦鏄撴棩鍒嗙墖鑾峰彇娑ㄨ穼鍋滅粺璁★紙limit_list锛?        杩斿洖锛氬皝鍗曢噾棰濄€佸皝鍗曢噺銆佹定鍋滅被鍨嬬瓑
        """
        trade_cal = self.get_trade_cal(start_date=start_date, end_date=end_date)
        if trade_cal is None or trade_cal.empty:
            return pd.DataFrame()

        all_dfs = []
        dates = trade_cal['cal_date'].tolist()
        print(f"鎸変氦鏄撴棩鍒嗙墖鑾峰彇娑ㄨ穼鍋滅粺璁★紙limit_list锛夛紝鍏?{len(dates)} 涓氦鏄撴棩...")
        cache_dir = os.path.join(config.CACHE_DIR, "limit_list")
        os.makedirs(cache_dir, exist_ok=True)

        for i, trade_date in enumerate(dates):
            try:
                cache_file = os.path.join(cache_dir, f"{trade_date}.csv")
                cache_exists = os.path.exists(cache_file)
                if cache_exists:
                    df_day = pd.read_csv(cache_file, dtype={'trade_date': str})
                else:
                    df_day = self.pro.limit_list(trade_date=trade_date)
                    if df_day is not None and not df_day.empty:
                        df_day.to_csv(cache_file, index=False, encoding='utf-8-sig')
                if df_day is not None and not df_day.empty:
                    keep_cols = ['ts_code', 'trade_date', 'name', 'close', 'pct_chg',
                                 'fc_ratio', 'fd_amount', 'first_time', 'last_time',
                                 'open_times', 'strth', 'limit']
                    available = [c for c in keep_cols if c in df_day.columns]
                    all_dfs.append(df_day[available])
                if (i + 1) % 30 == 0 or (i + 1) == len(dates):
                    print(f"  limit_list fetched {i+1}/{len(dates)} dates, rows={sum(len(d) for d in all_dfs)}")
                if not cache_exists:
                    time.sleep(0.35)
            except Exception as e:
                print(f"  鑾峰彇 {trade_date} 娑ㄨ穼鍋滅粺璁″け璐? {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"limit_list merged: {len(df_all)} rows")
        return df_all

    def _get_all_stocks_daily_serial(self, stock_list, start_date=None, end_date=None):
        """Fetch data stock by stock as fallback."""
        all_data = {}
        total = len(stock_list)
        name_changes = None
        if getattr(config, 'USE_HISTORICAL_ST_FILTER', False):
            name_changes = self._get_name_changes()

        for idx, row in stock_list.iterrows():
            ts_code = row['ts_code']
            name = row.get('name', '')

            df = self.get_stock_daily(ts_code, start_date, end_date)

            if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
                try:
                    daily_basic = self.pro.daily_basic(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        fields=(
                            'ts_code,trade_date,turnover_rate,turnover_rate_f,'
                            'volume_ratio,pe,pe_ttm,pb,ps,total_mv,circ_mv'
                        )
                    )
                    if daily_basic is None or daily_basic.empty:
                        raise ValueError("缺少历史 daily_basic")
                    daily_basic['trade_date'] = daily_basic['trade_date'].astype(str)
                    df['trade_date'] = df['trade_date'].astype(str)
                    basic_cols = [
                        col for col in daily_basic.columns
                        if col not in ('ts_code', 'trade_date')
                    ]
                    df = df.merge(
                        daily_basic[['trade_date'] + basic_cols],
                        on='trade_date', how='left'
                    )
                    for col in basic_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').ffill()
                except Exception as e:
                    print(f"[跳过] {ts_code} {name}: 历史 daily_basic 获取失败 - {e}")
                    time.sleep(0.2)
                    continue

                meta = row.to_dict()
                df = self._apply_stock_metadata(df, meta, name_changes)
                all_data[ts_code] = df
                print(f"[{len(all_data)}/{total}] {ts_code} {name}: {len(df)} rows")
            else:
                print(f"[璺宠繃] {ts_code} {name}: 鏁版嵁涓嶈冻")

            time.sleep(0.2)  # 闄愬埗璇锋眰棰戠巼锛岄伩鍏嶈Е鍙?Tushare 闄愭祦

        print(f"鎴愬姛鑾峰彇 {len(all_data)} 鍙偂绁ㄧ殑鏁版嵁")
        return all_data

    def filter_by_liquidity(self, stock_data, trade_date):
        """
        鏍规嵁娴佸姩鎬х瓫閫夎偂绁紙鏀寔鐩樹腑杩愯鏃跺洖閫€鍒版渶鏂版湁鏁版嵁鏃ユ湡锛?
        Args:
            stock_data: 鑲＄エ鏁版嵁瀛楀吀
            trade_date: 浜ゆ槗鏃ユ湡

        Returns:
            dict: 绛涢€夊悗鐨勮偂绁ㄦ暟鎹?        """
        filtered_data = {}

        for ts_code, df in stock_data.items():
            if df.empty:
                continue

            day_data = df[df['trade_date'] == trade_date]

            # 鐩樹腑杩愯鏃讹紝褰撳ぉ鏁版嵁鍙兘灏氭湭鏇存柊锛屽洖閫€鍒版渶杩戞湁鏁版嵁鐨勪氦鏄撴棩
            if day_data.empty:
                if getattr(config, 'STRICT_PREDICT_TRADE_DATE', True):
                    continue
                latest_date = df['trade_date'].max()
                day_data = df[df['trade_date'] == latest_date]
                if day_data.empty:
                    continue

            row = day_data.iloc[0]

            if row['vol'] < config.MIN_VOLUME:
                continue
            if row['amount'] < config.MIN_AMOUNT:
                continue

            if config.EXCLUDE_LIMIT_UP and row['pct_chg'] > config.LIMIT_THRESHOLD * 100:
                continue
            if config.EXCLUDE_LIMIT_DOWN and row['pct_chg'] < -config.LIMIT_THRESHOLD * 100:
                continue

            filtered_data[ts_code] = df

        print(f"Liquidity filter kept {len(filtered_data)} stocks")
        return filtered_data

    def calculate_volatility(self, df, period=None):
        """
        璁＄畻娉㈠姩鐜?
        Args:
            df: 鏃ョ嚎鏁版嵁
            period: 璁＄畻鍛ㄦ湡

        Returns:
            float: 娉㈠姩鐜?        """
        if period is None:
            period = config.MAX_VOLATILITY_DAYS

        if len(df) < period:
            return np.nan

        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(period).std()

        return volatility

    def filter_by_volatility(self, stock_data):
        """
        鏍规嵁娉㈠姩鐜囩瓫閫夎偂绁?
        Args:
            stock_data: 鑲＄エ鏁版嵁瀛楀吀

        Returns:
            dict: 绛涢€夊悗鐨勮偂绁ㄦ暟鎹?        """
        filtered_data = {}

        for ts_code, df in stock_data.items():
            volatility = self.calculate_volatility(df)

            if np.isnan(volatility) or volatility > config.MAX_VOLATILITY_THRESHOLD:
                continue

            filtered_data[ts_code] = df

        print(f"Volatility filter kept {len(filtered_data)} stocks")
        return filtered_data


def main():
    """Test data loader."""
    loader = DataLoader()

    # 鑾峰彇鏈€杩戜氦鏄撴棩
    trade_date = loader.get_latest_trade_date()
    print(f"鏈€杩戜氦鏄撴棩: {trade_date}")

    # 鑾峰彇鑲＄エ鍒楄〃
    stock_list = loader.get_stock_list(trade_date)
    print(f"Stock list size: {len(stock_list)}")

    # 娴嬭瘯鑾峰彇鍗曞彧鑲＄エ鏁版嵁
    if not stock_list.empty:
        test_code = stock_list.iloc[0]['ts_code']
        test_name = stock_list.iloc[0]['name']
        df = loader.get_stock_daily(test_code)
        if df is not None:
            print(f"\n{test_code} {test_name} 鏁版嵁鏍蜂緥:")
            print(df.tail())


if __name__ == "__main__":
    main()

