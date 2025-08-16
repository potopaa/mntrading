import pandas as pd

def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    fee_rate: float = 0.0005
) -> pd.DataFrame:
    """
    Vectorized backtest for a trading strategy.

    Parameters:
    - signals: pd.Series of position signals (-1, 0, +1), indexed by timestamp.
    - prices: pd.Series of close prices, same index as signals.
    - fee_rate: commission rate per transaction.

    Returns:
    - pd.DataFrame with columns:
        'returns',   # asset returns per period
        'pnl',       # profit & loss of strategy per period
        'turnover',  # absolute change in position (for cost calc)
        'costs',     # transaction costs per period
        'equity',    # cumulative equity curve
        'drawdown'   # drawdown relative to peak equity
    """
    # 1. Compute asset returns
    returns = prices.pct_change().fillna(0)

    # 2. Align signals and avoid lookahead bias
    positions = signals.shift(1).fillna(0)

    # 3. Strategy P&L: position * asset return
    pnl = positions * returns

    # 4. Turnover: absolute position change
    turnover = (positions - positions.shift(1)).abs().fillna(0)

    # 5. Transaction costs
    costs = turnover * fee_rate

    # 6. Equity curve: cumulative product of (1 + pnl - costs)
    equity = (1 + pnl - costs).cumprod()

    # 7. Drawdown: drop from running peak
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # 8. Assemble results
    result = pd.DataFrame({
        "returns": returns,
        "pnl": pnl,
        "turnover": turnover,
        "costs": costs,
        "equity": equity,
        "drawdown": drawdown
    })

    return result
