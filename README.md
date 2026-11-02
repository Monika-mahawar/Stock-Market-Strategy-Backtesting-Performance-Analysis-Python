# Stock Market Strategy Backtesting & Performance Analysis

## Project Overview
This project develops a Python-based stock market backtesting system using historical ICICI Bank (NSE) stock data from 2000 to 2021. It implements and evaluates two algorithmic trading strategies and compares their performance against a traditional Buy-and-Hold investment approach.

## Objectives
- Implement technical indicator-based trading strategies.
- Evaluate historical trading performance.
- Analyse investment returns and associated risks.
- Compare automated trading strategies with Buy-and-Hold benchmarks.

## Trading Strategies

### 1. Moving Average Crossover (20/50)
- Uses 20-day and 50-day Simple Moving Averages.
- Generates a buy signal when the short-term moving average exceeds the long-term moving average.
- Follows a trend-based trading approach.

### 2. RSI Mean Reversion
- Uses the 14-day Relative Strength Index (RSI).
- Enters a position when RSI falls below 30.
- Exits when RSI rises above 55.
- Follows a mean-reversion trading approach.

## Data Processing
- Historical ICICI Bank stock data (OHLCV).
- Handles missing values and duplicate dates.
- Adjusts historical prices for stock splits and bonus issues.
- Incorporates transaction costs of 0.07% per trade.

## Performance Evaluation

The strategies are evaluated using the following financial metrics:

| Metric | Description |
|---|---|
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Risk-adjusted return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Total Return | Overall investment return |
| Number of Trades | Total completed trading positions |

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib

## Project Structure

```text
Stock-Market-Strategy-Backtesting/
│
├── backtest.py
├── ICICIBANK.csv
├── equity_curve.png
├── backtest_results_ma_crossover.csv
├── backtest_results_rsi_meanrev.csv
└── README.md
```

## Installation & Execution

### 1. Clone the repository

```bash
git clone https://github.com/Monika-mahawar/Stock-Market-Strategy-Backtesting-Performance-Analysis-Python.git
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib
```

### 3. Run the backtesting system

Make sure `backtest.py` and `ICICIBANK.csv` are in the same directory.

```bash
python backtest.py
```

The script evaluates both strategies, prints performance metrics, exports individual backtesting results as CSV files, and generates an equity curve chart.

## Results & Visualization

The project generates:
- Strategy-wise performance comparison.
- Buy-and-Hold benchmark comparison.
- Equity curve visualization.
- CSV files containing individual strategy results.

## Key Learning Outcomes
- Understanding algorithmic trading strategies.
- Applying Python to financial data analysis.
- Implementing historical backtesting methodologies.
- Evaluating risk-adjusted investment performance.
- Analysing financial time-series data.

## Disclaimer
This project is developed for educational and analytical purposes only. Historical backtesting results do not guarantee future investment performance. It does not constitute financial or investment advice.
