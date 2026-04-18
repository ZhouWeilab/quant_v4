# quant_v4
quant strategy owned by zw
Here's a concise English overview of your project:

---

**Project Overview: A-Share Ultra-Short-Term Quantitative Trading System**

This project is an end-to-end quantitative trading system designed for China’s A‑share market, focusing on ultra‑short‑term strategies (buy today, sell tomorrow). It employs a hybrid deep learning architecture combining **CNN**, **Bi‑LSTM/GRU**, and **multi‑head self‑attention** to predict the next‑day return of individual stocks. The model takes a 20‑day sliding window of over 80 technical features (price, volume, moving averages, RSI, MACD, Bollinger Bands, ATR, etc.) and outputs a forecasted 1‑day ahead return. A custom composite loss function—integrating MSE, directional accuracy, and ranking loss—optimizes the model to prioritise correct direction and relative ordering among stocks, which is critical for high‑noise, weak‑signal A‑share environments.

The system is fully modular, with separate components for data loading (Tushare API with local caching to avoid rate limits), feature engineering, dataset construction (RobustScaler, temporal splitting), model training (early stopping, learning rate decay, checkpointing), and daily prediction. Real‑time filters exclude ST stocks, delisted shares, suspension, newly listed stocks, illiquid assets, high‑volatility securities, and limit‑up/down stocks to ensure tradability. Every day, the top 10 stocks with the highest predicted returns are recommended. A built‑in backtesting engine validates historical performance using metrics such as direction accuracy, information coefficient (IC), Sharpe ratio, and maximum drawdown. The entire pipeline is implemented in Python with TensorFlow/Keras, Pandas, and Scikit‑learn, and can be executed via a command‑line interface (`train`, `predict`, `backtest` modes). This system demonstrates how advanced deep learning techniques can be applied to extract faint predictive signals from noisy financial time series under strict T+1 settlement rules.
