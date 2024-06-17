# Agrega las importaciones necesarias y define las funciones
import tkinter as tk
from tkinter import ttk
import threading
import websocket
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

# Variables globales
current_symbol = None
current_interval = '1m'
ws = None
price_data = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
crypto_decimals = {}
model = None
scaler = None

def get_symbol_decimals():
    global crypto_decimals
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    response = requests.get(url)
    data = response.json()
    for symbol_info in data['symbols']:
        symbol = symbol_info['symbol']
        min_price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', symbol_info['filters']), None)
        if min_price_filter:
            tick_size = float(min_price_filter['tickSize'])
            decimals = '{:.8f}'.format(tick_size).rstrip('0').split('.')[1]
            crypto_decimals[symbol] = len(decimals)

def format_price(symbol, price):
    decimals = crypto_decimals.get(symbol, 2)
    return f"{price:.{decimals}f}"

def on_message(ws, message):
    global price_data
    try:
        data = json.loads(message)
        if 'e' in data and data['e'] == 'kline' and data['s'] == current_symbol:
            kline = data['k']
            if kline['i'] == current_interval:
                time = datetime.fromtimestamp(kline['t'] / 1000)
                open_price = float(kline['o'])
                high_price = float(kline['h'])
                low_price = float(kline['l'])
                close_price = float(kline['c'])
                new_data = pd.DataFrame([[time, open_price, high_price, low_price, close_price]], columns=["time", "open", "high", "low", "close"])
                price_data = pd.concat([price_data, new_data]).drop_duplicates(subset=['time']).tail(500)
                update_plot()
                lbl_price.config(text=f"Último precio: {format_price(current_symbol, close_price)}")
    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")

def subscribe_to_symbol(symbol, interval):
    if ws and ws.sock and ws.sock.connected:
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [
                f"{symbol.lower()}@kline_{interval}"
            ],
            "id": 1
        }))
        print(f"Suscrito a {symbol} con intervalo {interval}")

def unsubscribe_from_symbol(symbol, interval):
    if ws and ws.sock and ws.sock.connected:
        ws.send(json.dumps({
            "method": "UNSUBSCRIBE",
            "params": [
                f"{symbol.lower()}@kline_{interval}"
            ],
            "id": 1
        }))
        print(f"Cancelada la suscripción a {symbol} con intervalo {interval}")

def fetch_historical_data(symbol, interval):
    base_url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 500
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    historical_data = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    historical_data["time"] = pd.to_datetime(historical_data["time"], unit='ms')
    historical_data["open"] = historical_data["open"].astype(float)
    historical_data["high"] = historical_data["high"].astype(float)
    historical_data["low"] = historical_data["low"].astype(float)
    historical_data["close"] = historical_data["close"].astype(float)
    return historical_data[["time", "open", "high", "low", "close"]]

def prepare_data(df, window_size):
    data = []
    labels = []
    for i in range(len(df) - window_size):
        data.append(df.iloc[i:i + window_size].values)
        labels.append(df.iloc[i + window_size].values)
    return np.array(data), np.array(labels)

def train_model(data):
    global model, scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    window_size = 60
    X, y = prepare_data(pd.DataFrame(scaled_data), window_size)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, data.shape[1])))
    model.add(LSTM(50))
    model.add(Dense(data.shape[1]))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)

def on_select(event):
    global current_symbol, price_data, model, scaler
    selection = lb.curselection()
    if selection:
        new_symbol = lb.get(selection)
        if current_symbol:
            unsubscribe_from_symbol(current_symbol, current_interval)
        current_symbol = new_symbol
        price_data = fetch_historical_data(current_symbol, current_interval)
        subscribe_to_symbol(current_symbol, current_interval)
        lbl_price.config(text=f"Datos históricos cargados para {current_symbol}")
        
        # Entrena el modelo con los nuevos datos
        train_model(price_data[["open", "high", "low", "close"]].values)

def on_interval_change(event):
    global current_interval, price_data, model, scaler
    new_interval = interval_var.get()
    if current_symbol and new_interval != current_interval:
        unsubscribe_from_symbol(current_symbol, current_interval)
        current_interval = new_interval
        price_data = fetch_historical_data(current_symbol, current_interval)
        subscribe_to_symbol(current_symbol, current_interval)
        
        # Entrena el modelo con los nuevos datos
        train_model(price_data[["open", "high", "low", "close"]].values)

def update_list(*args):
    search_term = search_var.get().upper()
    lb.delete(0, tk.END)
    for symbol in filter(lambda x: search_term in x, crypto_list):
        lb.insert(tk.END, symbol)

def load_crypto_list():
    global crypto_list
    crypto_list = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
        'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'SHIBUSDT'
    ]
    for symbol in crypto_list:
        lb.insert(tk.END, symbol)

def run_websocket():
    global ws
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws",
        on_message=on_message,
        on_error=lambda ws, error: print(f"WebSocket error: {error}"),
        on_close=lambda ws, close_status_code, close_msg: print("WebSocket closed")
    )
    ws.run_forever()

def update_plot():
    if not price_data.empty:
        ax.clear()
        mpf.plot(price_data.set_index('time'), type='candle', ax=ax, style='charles')
        ax.set_title(f"{current_symbol} Price")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.tight_layout()
        canvas.draw()

def suggest_position():
    global model, scaler, price_data
    
    if len(price_data) < 60:
        return "Insufficient data for prediction"
    
    latest_data = scaler.transform(price_data.tail(60)[["open", "high", "low", "close"]])
    latest_data = np.expand_dims(latest_data, axis=0)
    
    prediction = model.predict(latest_data)
    suggestion = scaler.inverse_transform(prediction)[0]
    
    entry_price = suggestion[3]
    take_profit = entry_price * 1.02
    stop_loss = entry_price * 0.98
    
    return f"Entrada: {format_price(current_symbol, entry_price)}, TP: {format_price(current_symbol, take_profit)}, SL: {format_price(current_symbol, stop_loss)}"

def on_suggest():
    suggestion = suggest_position()
    lbl_suggestion.config(text=suggestion)

root = tk.Tk()
root.title("Crypto Price Viewer")

root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")

paned_window = tk.PanedWindow(root, orient=tk.VERTICAL)
paned_window.pack(fill=tk.BOTH, expand=True)

top_frame = tk.Frame(paned_window)
paned_window.add(top_frame)

bottom_frame = tk.Frame(paned_window)
paned_window.add(bottom_frame)

fig_width = 10
fig_height = 5

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
canvas = FigureCanvasTkAgg(fig, master=top_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

control_frame = tk.Frame(bottom_frame)
control_frame.pack(fill=tk.BOTH, expand=True)

lbl_price = ttk.Label(control_frame, text="Seleccione una criptomoneda", font=("Helvetica", 16))
lbl_price.pack(pady=10)

interval_var = tk.StringVar()
interval_combo = ttk.Combobox(control_frame, textvariable=interval_var, values=['1s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'])
interval_combo.current(1)  # Selecciona '1m' por defecto
interval_combo.pack(pady=5)
interval_combo.bind('<<ComboboxSelected>>', on_interval_change)

search_var = tk.StringVar()
search_var.trace("w", update_list)

search_entry = ttk.Entry(control_frame, textvariable=search_var)
search_entry.pack(fill=tk.X, padx=5, pady=5)

lb = tk.Listbox(control_frame, height=6)
lb.pack(fill=tk.BOTH, expand=True)
lb.bind('<<ListboxSelect>>', on_select)

scrollbar = tk.Scrollbar(control_frame, orient="vertical")
scrollbar.config(command=lb.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

lb.config(yscrollcommand=scrollbar.set)

btn_suggest = ttk.Button(control_frame, text="Hack Position", command=on_suggest)
btn_suggest.pack(pady=5)

lbl_suggestion = ttk.Label(control_frame, text="", font=("Helvetica", 16))
lbl_suggestion.pack(pady=10)

load_crypto_list()
get_symbol_decimals()

threading.Thread(target=run_websocket, daemon=True).start()

root.mainloop()
