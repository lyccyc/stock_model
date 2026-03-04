import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

def load_latest_market_data():
    """
    Loads predicting/historic LSTM data, moving averages and current actuals.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'taiwan_stock_processed.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing historical data: {data_path}")
        
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    if df.empty:
        raise ValueError("Stock data is empty.")
        
    # Get the latest available row for context
    latest_data = df.iloc[-1]
    
    # Load real LSTM prediction if available
    prediction_path = os.path.join(project_root, 'data', 'latest_prediction.json')
    lstm_predicted_price = float(latest_data['Close']) * 1.02
    if os.path.exists(prediction_path):
        with open(prediction_path, 'r') as f:
            pred_data = json.load(f)
            if 'predicted_price' in pred_data:
                lstm_predicted_price = float(pred_data['predicted_price'])
    
    return {
        'date': str(latest_data.name.date()),
        'current_price': float(latest_data['Close']),
        'ma20': float(latest_data['MA20']),
        'rsi': float(latest_data['RSI14']),
        'lstm_predicted_price': lstm_predicted_price
    }

def load_sentiment_data():
    """
    Loads daily composite sentiment score mapped by FinBERT.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentiment_path = os.path.join(project_root, 'data', 'news_sentiment_results.csv')
    
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"Missing sentiment data: {sentiment_path}")
        
    df = pd.read_csv(sentiment_path)
    if df.empty:
        raise ValueError("Sentiment data is empty.")
        
    latest_avg_sentiment = df['sentiment_index'].mean()
    return float(latest_avg_sentiment)

def generate_decision(market_data, avg_sentiment):
    """
    Makes a professional strategy call via Gemini using all variables.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
        
    client = genai.Client(api_key=api_key)
    
    system_prompt = """You are the Chief Investment Strategist for an advanced systematic trading fund.
Your job is to fuse quantitative LSTM algorithmic price predictions with complex unstructured NLP News Sentiment.

Decision Rules:
1. If LSTM is highly bullish BUT Sentiment is extremely bearish (< -0.3), lean towards caution (HOLD or SELL).
2. If both LSTM and Sentiment agree on the trend, exhibit strong confidence (STRONG_BUY or SELL).
3. If RSI is > 70 (Overbought) and Sentiment is cooling, consider taking profits.

You MUST respond strictly with a valid JSON document (no markdown formatting or codeblocks) with the following structure:
{
    "action": "BUY | SELL | HOLD | STRONG_BUY",
    "confidence_score": <float between 0.0 and 1.0>,
    "reasoning": "<short explanation of the data synthesis>",
    "risk_warning": "<brief warning if signals diverge>"
}
"""

    user_prompt = f"""Daily Data Synthesis for {market_data['date']}:
- Current Close Price: {market_data['current_price']:.2f}
- 20-Day Moving Average (MA20): {market_data['ma20']:.2f}
- RSI (14-Day): {market_data['rsi']:.2f}
- LSTM Predicted Next Day Price: {market_data['lstm_predicted_price']:.2f}
- NLP News Sentiment Index (-1.0 to 1.0): {avg_sentiment:.3f}

Provide your investment decision strictly adhering to the requested JSON format."""

    print("Requesting Gemini for final investment strategy...")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_prompt, user_prompt],
        )
        
        # Clean any accidental markdown artifacts around the json response
        response_text = response.text.strip().removeprefix('```json').removeprefix('```').removesuffix('```')
        decision_json = json.loads(response_text)
        return decision_json
        
    except Exception as e:
        print(f"Failed to generate decision from LLM: {str(e)}")
        # Fallback decision logic if API fails
        return {
            "action": "HOLD",
            "confidence_score": 0.0,
            "reasoning": f"System runtime Exception using API: {str(e)}",
            "risk_warning": "No valid LLM response. Holding to preserve capital."
        }

def process_strategy():
    try:
        print("Gathering quant trading metrics...")
        market = load_latest_market_data()
        sentiment = load_sentiment_data()
        
        print(f"\n[Data Feed] Current Price: {market['current_price']:.2f} | MA20: {market['ma20']:.2f} | RSI: {market['rsi']:.2f}")
        print(f"[Data Feed] LSTM Predicted (T+1): {market['lstm_predicted_price']:.2f}")
        print(f"[Data Feed] News Sentiment Index: {sentiment:.3f}")
        
    except Exception as e:
        print(f"Data validation error: {str(e)}")
        return

    # Call Gemini inference
    decision = generate_decision(market, sentiment)
    
    # Extract decision format
    action = decision.get("action", "HOLD")
    confidence = decision.get("confidence_score", 0.0)
    reasoning = decision.get("reasoning", "No valid explanation generated.")
    risk = decision.get("risk_warning", "None.")
    
    # Formatted Execution Logger
    report_content = f"""==================================
      DAILY INVESTMENT REPORT     
==================================
Date: {market['date']}
Action Rec: {action} (Confidence: {confidence:.2f})
Reasoning : {reasoning}
Risks     : {risk}
Technical Signal (LSTM Predicted T+1): {market['lstm_predicted_price']:.2f}
Sentiment Signal (FinBERT Index): {sentiment:.3f}
==================================
"""
    print(report_content)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_path = os.path.join(project_root, 'logs', 'trading_decisions.log')
    
    with open(logs_path, 'a') as f:
        log_entry = f"[{market['date']}] Action: {action} | Confidence: {confidence:.2f} | Price: {market['current_price']:.2f} | Sentiment: {sentiment:.3f} | Reason: {reasoning}\n"
        f.write(log_entry)
        
    report_path = os.path.join(project_root, 'logs', 'daily_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_content)
        
    print(f"Strategy securely logged to: {logs_path}")
    print(f"Summary report generated at: {report_path}")

if __name__ == "__main__":
    process_strategy()
