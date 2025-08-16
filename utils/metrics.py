import numpy as np
import pandas as pd


def cumulative_return(equity: pd.Series) -> float:
    """
    Compute cumulative return of an equity curve.

    Parameters:
    - equity: pd.Series of cumulative equity values (e.g. starting at 1.0)

    Returns:
    - Cumulative return, i.e. final_equity - 1
    """
    return equity.iloc[-1] - 1.0


def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio of a returns series.

    Parameters:
    - returns: pd.Series of period returns (not cumulative), e.g. daily PnL.
    - risk_free_rate: annual risk-free rate, default 0.
    - periods_per_year: number of trading periods per year (252 trading days).

    Returns:
    - Annualized Sharpe ratio.
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)

    if std_excess == 0:
        return np.nan

    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return sharpe


def max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown of an equity curve.

    Parameters:
    - equity: pd.Series of cumulative equity values.

    Returns:
    - Maximum drawdown as a negative float (e.g. -0.2 means 20% drawdown).
    """
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    return drawdowns.min()
