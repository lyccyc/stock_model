# AI Stock Analysis Robot for Taiwan 50 ETF (0050.TW)

An AI-driven quantitative and qualitative stock analysis tool specifically tailored for the Yuanta Taiwan 50 ETF (0050.TW) and its market context on the Taiwan Stock Exchange (TWSE).

This robot collects historical stock data, scrapes real-time news headlines, performs multi-lingual NLP sentiment analysis using FinBERT, predicts the next day's closing price with an LSTM neural network, and ultimately synthesizes all indicators into a daily trading decision (BUY/SELL/HOLD) using Google's Gemini Large Language Model.

## Features

- **Data Collection (`data_collector.py`)**: Fetches historical pricing and volume data via Yahoo Finance. Integrates TAIEX (`^TWII`) as a reference. Computes key technical indicators such as Moving Averages (MA5, MA10, MA20, MA60) and Relative Strength Index (RSI14).
- **News Scraping (`news_collector.py`)**: Automatically monitors Google News RSS feeds for the latest developments related to the 0050 ETF and the broader Taiwan stock market over the past 24 hours.
- **Sentiment Analysis (`sentiment_analyzer.py`)**: Translates Traditional Chinese market news to English using the `Helsinki-NLP/opus-mt-zh-en` model, then scores the market sentiment with the `ProsusAI/finbert` financial NLP model, calculating a daily composite sentiment index (-1.0 to 1.0).
- **Deep Learning Prediction (`train_lstm.py`)**: Trains a bidirectional-like LSTM (Long Short-Term Memory) sequential neural network on technical features (Close, Volume, MA5, MA20, RSI14) to forecast the ETF's next-day underlying closing price.
- **AI Decision Agent (`final_decision_agent.py`)**: Fuses the quantitative signals (LSTM predicted price, MA20, RSI) with unstructured qualitative signals (FinBERT Sentiment Index) by querying the Google Gemini API (gemini-2.5-flash). Generates an actionable investment report detailing reasoning, risk warnings, and the final strategy.

## Project Structure

```text
stock_model/
├── data/               # Contains active datasets (historical features, news, sentiment, predictions)
├── logs/               # Output directory for daily trading decision reports and metrics
├── models/             # Contains the trained `.h5` LSTM model and `scaler.pkl`
├── src/
│   ├── data_collector.py        # Fetches price and market features
│   ├── news_collector.py        # Scrapes latest market headlines
│   ├── sentiment_analyzer.py    # Generates translation and sentiment index
│   ├── train_lstm.py            # Model training and daily price prediction
│   └── final_decision_agent.py  # LLM (Gemini) strategy formulation
├── requirements.txt    # Python dependencies
├── .env                # API Keys (e.g. GEMINI_API_KEY)
└── README.md
```

## Setup & Requirements

1. Ensure you have Python 3.8+ installed.
2. Clone or navigate into this repository.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. You will need a functioning Gemini API Key for the `final_decision_agent.py` script. Create a `.env` file in the root directory:

   ```bash
   echo "GEMINI_API_KEY=your_genai_api_key_here" > .env
   ```

## Usage

For a complete daily run, execute the scripts sequentially:

1. **Update Technical Data**: `python src/data_collector.py`
2. **Fetch News Headlines**: `python src/news_collector.py`
3. **Analyze News Sentiment**: `python src/sentiment_analyzer.py`
4. **Train/Predict Prices**: `python src/train_lstm.py`
5. **Generate Final AI Decision**: `python src/final_decision_agent.py`

When the pipeline finishes, review the summary output dumped in the console, or find the archived text report located at `logs/daily_report.txt` and `logs/trading_decisions.log`.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. AI models, trading algorithms, and language model-generated investment decisions are volatile and experimental. Real capital should not be risked solely based on these automated strategies.
