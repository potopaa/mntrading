# -*- coding: utf-8 -*-
"""
spread.py — утилиты для построения спреда и скользящих признаков.
Содержит:
- compute_spread(df, a, b): базовый спред для совместимости (лог-спред по закрытиям)
- rolling_ols(y, x, window, min_periods)
- rolling_zscore(s, window, min_periods)
- НОВОЕ: build_close_matrix, compute_pair_features_from_prices, compute_features_for_pairs
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------ базовые функции (совместимость) ------------------------------ #

def _coerce_close_wide(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """
    Приводит входной df к wide-матрице закрытий: колонки — символы, индекс — datetime.
    Пытается распознать 3 формата:
      1) MultiIndex columns: (symbol, field) и поле 'close';
      2) Long-формат с колонками ['symbol','close'] и индексом-таймом или колонкой 'timestamp';
      3) Уже wide, где колонки — символы.
    """
    # 1) MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        names = df.columns.names or []
        last_level = names[-1] if names else None
        if last_level:
            # попробуем срез по последнему уровню
            try:
                wide = df.xs('close', axis=1, level=last_level)
                return wide[[a, b]]
            except Exception:
                pass
        if df.columns.nlevels >= 2:
            try:
                wide = df.xs('close', axis=1, level=1)
                return wide[[a, b]]
            except Exception:
                pass

    # 2) Long-формат
    cols_lower = {c.lower(): c for c in df.columns}
    if 'symbol' in cols_lower and 'close' in cols_lower:
        # индекс — уже datetime?
        if not isinstance(df.index, pd.DatetimeIndex):
            ts_col = cols_lower.get('timestamp')
            if ts_col is not None:
                idx = pd.to_datetime(df[ts_col], unit='ms', errors='coerce')
            else:
                idx = pd.to_datetime(df.index, errors='coerce')
            df = df.copy()
            df.index = idx
        tmp = df[[cols_lower['symbol'], cols_lower['close']]].copy()
        tmp['__ts__'] = df.index
        wide = tmp.pivot_table(index='__ts__', columns=cols_lower['symbol'],
                               values=cols_lower['close'], aggfunc='last')
        return wide[[a, b]]

    # 3) fallback — считаем, что df уже wide: колонки — символы
    return df[[a, b]].copy()


def compute_spread(raw_df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """
    Совместимая реализация базового спреда:
      spread = log(close_a) - log(close_b)
    """
    px = _coerce_close_wide(raw_df, a, b).dropna()
    px.columns = ['pa', 'pb']
    spread = np.log(px['pa']) - np.log(px['pb'])
    spread.name = f"spread_{a}__{b}"
    return spread


def rolling_ols(y: pd.Series, x: pd.Series, window: int = 300, min_periods: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS: y = alpha + beta * x.
    Возвращает (beta, alpha) как Series, выровненные по индексу.
    """
    if min_periods is None:
        min_periods = window

    mean_x = x.rolling(window, min_periods=min_periods).mean()
    mean_y = y.rolling(window, min_periods=min_periods).mean()
    var_x  = x.rolling(window, min_periods=min_periods).var(ddof=0)
    cov_xy = (x * y).rolling(window, min_periods=min_periods).mean() - mean_x * mean_y

    beta = cov_xy / var_x.replace(0, np.nan)
    alpha = mean_y - beta * mean_x
    beta.name = "beta"
    alpha.name = "alpha"
    return beta, alpha


def rolling_zscore(s: pd.Series, window: int = 300, min_periods: Optional[int] = None) -> pd.Series:
    """
    Скользящий z-score.
    """
    if min_periods is None:
        min_periods = window
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    z.name = "z"
    return z


# ------------------------------ расширенные функции (мультипары) ------------------------------ #

def build_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Унифицирует сырой слепок OHLCV в wide-матрицу закрытий (index=datetime, columns=symbol -> close).
    Поддерживает:
      1) MultiIndex columns: (symbol, field) и поле 'close'
      2) Long-формат с колонками ['symbol','close'] и индексом-таймом (или колонкой 'timestamp')
      3) Уже wide-формат (fallback)
    """
    # 1) MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.names[-1] if df.columns.names else None
        if last_level is not None:
            try:
                wide = df.xs('close', axis=1, level=last_level)
                return wide.sort_index()
            except (KeyError, ValueError):
                pass
        if df.columns.nlevels >= 2:
            try:
                wide = df.xs('close', axis=1, level=1)
                return wide.sort_index()
            except (KeyError, ValueError):
                pass

    # 2) Long-формат
    cols_lower = {c.lower(): c for c in df.columns}
    if 'symbol' in cols_lower and 'close' in cols_lower:
        if not isinstance(df.index, pd.DatetimeIndex):
            ts_col = cols_lower.get('timestamp')
            if ts_col is not None:
                idx = pd.to_datetime(df[ts_col], unit='ms', errors='coerce')
            else:
                idx = pd.to_datetime(df.index, errors='coerce')
            df = df.copy()
            df.index = idx
        tmp = df[[cols_lower['symbol'], cols_lower['close']]].copy()
        tmp['__ts__'] = df.index
        wide = tmp.pivot_table(index='__ts__', columns=cols_lower['symbol'],
                               values=cols_lower['close'], aggfunc='last')
        wide.index.name = None
        return wide.sort_index()

    # 3) fallback — уже wide
    return df.copy().sort_index()


def _rolling_ols_y_on_x(y: pd.Series, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Rolling OLS: y = alpha + beta * x (скользящее окно).
    Возвращает DataFrame ['beta','alpha'].
    """
    if min_periods is None:
        min_periods = window

    mean_x = x.rolling(window, min_periods=min_periods).mean()
    mean_y = y.rolling(window, min_periods=min_periods).mean()
    var_x  = x.rolling(window, min_periods=min_periods).var(ddof=0)
    cov_xy = (x * y).rolling(window, min_periods=min_periods).mean() - mean_x * mean_y

    beta = cov_xy / var_x.replace(0, np.nan)
    alpha = mean_y - beta * mean_x

    out = pd.DataFrame({'beta': beta, 'alpha': alpha})
    return out


def _rolling_zscore(s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def compute_pair_features_from_prices(
    px: pd.DataFrame,
    a: str,
    b: str,
    beta_window: int = 300,
    z_window: int = 300,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Для пары (a,b) строит фичи на основе цен закрытия: beta, alpha (rolling OLS), spread, z.
    px: wide-матрица закрытий (index=datetime, columns=symbol).
    """
    if a not in px.columns or b not in px.columns:
        return pd.DataFrame()

    df = px[[a, b]].dropna().copy()
    df.columns = ['pa', 'pb']

    ols = _rolling_ols_y_on_x(df['pa'], df['pb'], window=beta_window, min_periods=min_periods)
    spread = df['pa'] - (ols['alpha'] + ols['beta'] * df['pb'])
    z = _rolling_zscore(spread, window=z_window, min_periods=min_periods)

    out = pd.concat([df, ols, spread.rename('spread'), z.rename('z')], axis=1).dropna()
    out['a'] = a
    out['b'] = b
    out['pair'] = f'{a}__{b}'
    return out


def compute_features_for_pairs(
    raw_df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    beta_window: int = 300,
    z_window: int = 300,
    min_periods: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Строит фичи для нескольких пар. Возвращает словарь { 'A__B': df_features }.
    """
    px = build_close_matrix(raw_df)
    results: Dict[str, pd.DataFrame] = {}
    for (a, b) in pairs:
        feat = compute_pair_features_from_prices(px, a, b, beta_window, z_window, min_periods)
        key = f'{a}__{b}'
        if not feat.empty:
            results[key] = feat
    return results
