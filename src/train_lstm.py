import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, target_col_idx, lookback_period=60):
    """
    Creates moving window sequences for LSTM input.
    """
    X, y = [], []
    # from lookback_period to end of dataset
    for i in range(lookback_period, len(data)):
        X.append(data[i-lookback_period:i]) # past 'lookback_period' days
        y.append(data[i, target_col_idx])   # predict next day's target
    return np.array(X), np.array(y)

def train_and_evaluate():
    # 1. Project directories setup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'taiwan_stock_processed.csv')
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
        
    # 1. Data Loading
    print("Loading data...")
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # Check if necessary columns exist
    # RSI was named 'RSI14' in data_collector.py
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI14']
    for f in features:
        if f not in df.columns:
            print(f"Missing required feature column: {f}")
            return
            
    # 2. Feature Engineering & Scaling
    print("Scaling features...")
    data = df[features].values
    target_col_idx = features.index('Close')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Sequence Creation
    lookback_period = 60
    print(f"Creating sequences with lookback period: {lookback_period} days...")
    X, y = create_sequences(scaled_data, target_col_idx, lookback_period)
    
    # Train/Test Split (80% / 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # 4. Model Architecture 
    print("Compiling LSTM model...")
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 5. Training Logic
    print("Training model with EarlyStopping...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
    
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'lstm_stock_model.h5')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    # Note: Keras recommends native format, but .h5 is perfectly valid (requires h5py)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Trained model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # 6. Evaluation & Plotting
    print("Evaluating test set predictions...")
    predictions_scaled = model.predict(X_test)
    
    # Transform predictions and actual y values back to original price scale
    def inverse_transform_target(scaled_target_values):
        # We need a dummy array of shape (N, features_len) to use the inverse scaler
        dummy = np.zeros((len(scaled_target_values), len(features)))
        dummy[:, target_col_idx] = scaled_target_values.flatten()
        return scaler.inverse_transform(dummy)[:, target_col_idx]
        
    y_test_inv = inverse_transform_target(y_test)
    predictions_inv = inverse_transform_target(predictions_scaled)
    
    test_dates = df.index[-len(y_test_inv):]
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, color='blue', label='Actual Price')
    plt.plot(test_dates, predictions_inv, color='red', label='Predicted Price')
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    plot_path = os.path.join(logs_dir, 'prediction_plot.png')
    
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Prediction comparison plot saved to: {plot_path}")

def predict_next_day():
    print("Generating prediction for the next trading day...")
    import json
    from tensorflow.keras.models import load_model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'taiwan_stock_processed.csv')
    model_path = os.path.join(project_root, 'models', 'lstm_stock_model.h5')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')

    if not os.path.exists(data_path) or not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Required files not found for prediction.")
        return

    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI14']
    data = df[features].values
    target_col_idx = features.index('Close')

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    # We need the last 60 days of data
    lookback_period = 60
    if len(data) < lookback_period:
        print("Not enough data to make a prediction.")
        return

    recent_data = data[-lookback_period:]
    scaled_recent = scaler.transform(recent_data)
    X_pred = np.array([scaled_recent])

    pred_scaled = model.predict(X_pred)

    dummy = np.zeros((1, len(features)))
    dummy[0, target_col_idx] = pred_scaled[0][0]
    pred_inv = scaler.inverse_transform(dummy)[0, target_col_idx]

    pred_result = {
        "date_extracted": str(df.index[-1].date()),
        "predicted_price": float(pred_inv)
    }

    output_path = os.path.join(project_root, 'data', 'latest_prediction.json')
    with open(output_path, 'w') as f:
        json.dump(pred_result, f, indent=4)
        
    print(f"Prediction saved to {output_path}: {pred_inv:.2f}")

if __name__ == "__main__":
    train_and_evaluate()
    predict_next_day()
