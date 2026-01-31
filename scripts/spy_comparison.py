"""Compare SPY performance to strategy across different periods."""
import yfinance as yf
import numpy as np
import pandas as pd

# Download SPY data with auto_adjust
spy = yf.download('SPY', start='2018-01-01', end='2026-01-01', progress=False, auto_adjust=True)
spy['Return'] = spy['Close'].pct_change()

def calc_sharpe(returns, annualize=252):
    returns = returns.dropna()
    if len(returns) < 20:
        return np.nan
    return np.sqrt(annualize) * returns.mean() / returns.std()

def calc_total_return(prices):
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100

def calc_max_dd(prices):
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    return dd.min() * 100

periods = [
    ('Fold 1 (2018-01 to 2020-06)', '2018-01-01', '2020-06-30'),
    ('Fold 2 (2020-01 to 2022-06)', '2020-01-01', '2022-06-30'),
    ('Fold 3 (2021-01 to 2023-12)', '2021-01-01', '2023-12-31'),
    ('Holdout (2024-01 to 2025-12)', '2024-01-01', '2025-12-31'),
    ('Bear Market 2022', '2022-01-01', '2022-12-31'),
]

print('='*70)
print('SPY PERFORMANCE BY PERIOD')
print('='*70)
print()
print(f"Period                              Sharpe      Return     Max DD")
print('-'*70)

spy_sharpes = {}
for name, start, end in periods:
    mask = (spy.index >= start) & (spy.index <= end)
    period_data = spy.loc[mask]
    
    sharpe = calc_sharpe(period_data['Return'])
    total_ret = float(calc_total_return(period_data['Close']))
    max_dd = float(calc_max_dd(period_data['Close']))
    
    short_name = name.split('(')[0].strip()
    spy_sharpes[short_name] = sharpe
    
    print(f"{name:<35} {sharpe:>8.2f} {total_ret:>10.1f}% {max_dd:>9.1f}%")

print()
print('='*70)
print('COMPARISON: Your Strategy vs SPY (Sharpe Ratio)')
print('='*70)
print()
print("Period                   Strategy         SPY      Excess")
print('-'*70)

strategy_sharpes = {
    'Fold 1': -0.08,
    'Fold 2': 3.36,
    'Fold 3': 0.89,
    'Holdout': 5.64,
}

for name, strat in strategy_sharpes.items():
    spy_s = spy_sharpes.get(name, np.nan)
    excess = strat - spy_s
    print(f"{name:<25} {strat:>10.2f} {spy_s:>10.2f} {excess:>+10.2f}")

print()
print('='*70)
print('INTERPRETATION')
print('='*70)
print("""
- If 'Excess' is positive, your strategy beat SPY on risk-adjusted basis
- If SPY Sharpe is also very high in a period, the market was favorable
- A Sharpe > 2.0 for SPY is unusually bullish market conditions
""")
