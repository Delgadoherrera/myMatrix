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
import mplfinance as mpf

# Variables globales
current_symbol = None
current_interval = '1m'
ws = None
price_data = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

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

def on_select(event):
    global current_symbol, price_data
    selection = lb.curselection()
    if selection:
        new_symbol = lb.get(selection)
        if current_symbol:
            unsubscribe_from_symbol(current_symbol, current_interval)
        current_symbol = new_symbol
        price_data = fetch_historical_data(current_symbol, current_interval)
        subscribe_to_symbol(current_symbol, current_interval)
        lbl_price.config(text=f"Datos históricos cargados para {current_symbol}")

def on_interval_change(event):
    global current_interval, price_data
    new_interval = interval_var.get()
    if current_symbol and new_interval != current_interval:
        unsubscribe_from_symbol(current_symbol, current_interval)
        current_interval = new_interval
        price_data = fetch_historical_data(current_symbol, current_interval)
        subscribe_to_symbol(current_symbol, current_interval)

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
        canvas.draw()

root = tk.Tk()
root.title("Crypto Price Viewer")

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

search_frame = tk.Frame(left_frame)
search_frame.pack(fill=tk.X, padx=5, pady=5)

search_var = tk.StringVar()
search_var.trace("w", update_list)

search_entry = ttk.Entry(search_frame, textvariable=search_var)
search_entry.pack(fill=tk.X, padx=5, pady=5)

lb = tk.Listbox(left_frame, height=15)
lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
lb.bind('<<ListboxSelect>>', on_select)

scrollbar = tk.Scrollbar(left_frame, orient="vertical")
scrollbar.config(command=lb.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

lb.config(yscrollcommand=scrollbar.set)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

lbl_price = ttk.Label(right_frame, text="Seleccione una criptomoneda", font=("Helvetica", 16))
lbl_price.pack(pady=20)

# Tamaño de la figura del gráfico
fig_width = 5
fig_height = 4

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.draw()  # Añadir esta línea para inicializar el lienzo
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

interval_var = tk.StringVar()
interval_combo = ttk.Combobox(right_frame, textvariable=interval_var, values=['1s','1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'])
interval_combo.current(0)  # Selecciona '1m' por defecto
interval_combo.pack(pady=10)
interval_combo.bind('<<ComboboxSelected>>', on_interval_change)

load_crypto_list()

threading.Thread(target=run_websocket, daemon=True).start()

root.mainloop()
