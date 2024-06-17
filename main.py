import tkinter as tk
from tkinter import ttk
import threading
import websocket
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import mplfinance as mpf
import time
import pyperclip

# Función para inicializar la ventana principal y los indicadores técnicos seleccionados
def initialize_window():
    global root, indicators_selected

    # Crear la ventana principal
    root = tk.Tk()
    root.title("Position hacker Crypto")

    # Variables para seleccionar los indicadores técnicos
    indicators_selected = {
        'sma': tk.BooleanVar(value=True),
        'ema': tk.BooleanVar(value=True),
        'rsi': tk.BooleanVar(value=True),
        'bollinger': tk.BooleanVar(value=True),
        'macd': tk.BooleanVar(value=True),
        'atr': tk.BooleanVar(value=True)
    }

# Variables globales
current_symbol = None
current_interval = '1m'
ws = None
price_data = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
crypto_decimals = {}
model = None
input_scaler = None
output_scaler = None
progress_window = None
progress_bar = None

# Inicializar la ventana principal
initialize_window()

# Función para obtener datos históricos de Binance
def fetch_binance_historical_data(symbol, interval, limit=1000):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_asset_volume", "number_of_trades",
                                     "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["time"] = pd.to_datetime(df["time"], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]

# Función para agregar indicadores técnicos
def add_technical_indicators(df):
    if indicators_selected['sma'].get():
        df['sma'] = ta.sma(df['close'], length=14).fillna(0)
    if indicators_selected['ema'].get():
        df['ema'] = ta.ema(df['close'], length=14).fillna(0)
    if indicators_selected['rsi'].get():
        df['rsi'] = ta.rsi(df['close'], length=14).fillna(0)
    if indicators_selected['bollinger'].get():
        bbands = ta.bbands(df['close'], length=20)
        df['bollinger_lower'] = bbands[f'BBL_20_2.0'].fillna(0)
        df['bollinger_middle'] = bbands[f'BBM_20_2.0'].fillna(0)
        df['bollinger_upper'] = bbands[f'BBU_20_2.0'].fillna(0)
    if indicators_selected['macd'].get():
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9'].fillna(0)
        df['macd_signal'] = macd['MACDs_12_26_9'].fillna(0)
        df['macd_hist'] = macd['MACDh_12_26_9'].fillna(0)
    if indicators_selected['atr'].get():
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(0)
    return df

# Función para preparar los datos
def prepare_data_with_indicators(df, window_size):
    df = df.fillna(0)  # Llena los valores NaN
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # Selecciona las columnas con indicadores técnicos basados en los indicadores seleccionados
    features = ['open', 'high', 'low', 'close', 'volume']
    if indicators_selected['sma'].get():
        features.append('sma')
    if indicators_selected['ema'].get():
        features.append('ema')
    if indicators_selected['rsi'].get():
        features.append('rsi')
    if indicators_selected['bollinger'].get():
        features.extend(['bollinger_lower', 'bollinger_middle', 'bollinger_upper'])
    if indicators_selected['macd'].get():
        features.extend(['macd', 'macd_signal', 'macd_hist'])
    if indicators_selected['atr'].get():
        features.append('atr')

    scaled_data = input_scaler.fit_transform(df[features])

    labels = np.zeros((len(df), 4))
    labels[:, 0] = df['close']
    labels[:, 1] = df['close'] * 1.02  # Take Profit para Long
    labels[:, 2] = df['close'] * 0.98  # Stop Loss para Long
    labels[:, 3] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 si se espera que suba, 0 si se espera que baje

    scaled_labels = output_scaler.fit_transform(labels[:, :3])

    data, y, direction = [], [], []
    for i in range(len(scaled_data) - window_size):
        data.append(scaled_data[i:i + window_size])
        y.append(scaled_labels[i + window_size])
        direction.append(labels[i + window_size, 3])
    return np.array(data), np.array(y), np.array(direction), input_scaler, output_scaler

# Función para entrenar el modelo
def train_model_with_indicators(X, y, direction):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = LSTM(50, return_sequences=True)(inputs)
    x = LSTM(50)(x)

    # Predicción de precios (entrada, TP, SL)
    price_output = Dense(3)(x)

    # Predicción de la dirección (subida o bajada)
    direction_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=[price_output, direction_output])
    
    model.compile(optimizer='adam', loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[1.0, 0.5])
    model.fit(X, [y, direction], epochs=10, batch_size=32)
    return model

# Función para formatear los precios
def format_price(symbol, price):
    decimals = crypto_decimals.get(symbol, 2)
    return f"{price:.{decimals}f}"

# Función para sugerir posición de trading
def suggest_trading_position(model, input_scaler, output_scaler, price_data, symbol):
    if len(price_data) < 60:
        return "Datos insuficientes para la predicción."
    
    try:
        features = ['open', 'high', 'low', 'close', 'volume']
        if indicators_selected['sma'].get():
            features.append('sma')
        if indicators_selected['ema'].get():
            features.append('ema')
        if indicators_selected['rsi'].get():
            features.append('rsi')
        if indicators_selected['bollinger'].get():
            features.extend(['bollinger_lower', 'bollinger_middle', 'bollinger_upper'])
        if indicators_selected['macd'].get():
            features.extend(['macd', 'macd_signal', 'macd_hist'])
        if indicators_selected['atr'].get():
            features.append('atr')

        latest_data = input_scaler.transform(price_data.tail(60)[features])
        latest_data = np.expand_dims(latest_data, axis=0)
        price_prediction, direction_prediction = model.predict(latest_data)

        entry_price, take_profit, stop_loss = output_scaler.inverse_transform(price_prediction)[0]
        direction = direction_prediction[0][0]

        current_price = price_data['close'].iloc[-1]

        # Validar precios
        if np.isnan(entry_price) or entry_price <= 0:
            entry_price = "No disponible"
        if np.isnan(take_profit) or take_profit <= 0:
            take_profit = "No disponible"
        if np.isnan(stop_loss) or stop_loss <= 0:
            stop_loss = "No disponible"

        # Determinar la dirección y ajustar TP/SL en consecuencia
        if direction >= 0.5:
            position_type = "Long"
            take_profit = format_price(symbol, take_profit)
            stop_loss = format_price(symbol, stop_loss)
        else:
            position_type = "Short"
            take_profit = format_price(symbol, current_price - (entry_price - current_price))
            stop_loss = format_price(symbol, current_price + (current_price - entry_price))
            entry_price = format_price(symbol, entry_price)

        return {
            'position_type': position_type,
            'entry_price': format_price(symbol, entry_price) if entry_price != "No disponible" else "No disponible",
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }
    except Exception as e:
        print(f"Error al sugerir posición de trading: {e}")
        return {
            'position_type': "N/A",
            'entry_price': "No disponible",
            'take_profit': "No disponible",
            'stop_loss': "No disponible"
        }

# Función para mostrar la recomendación de posición de trading
def show_trading_position(suggestion):
    position_window = tk.Toplevel(root)
    position_window.title("Recomendación de Trading")
    position_window.geometry("400x250")

    def copy_to_clipboard(text):
        root.clipboard_clear()  # Limpia el contenido del portapapeles
        root.clipboard_append(text)  # Añade el texto al portapapeles
        root.update()  # Actualiza el portapapeles para asegurarse de que el contenido esté disponible
        tk.messagebox.showinfo("Copiar", "Copiado al portapapeles")

    ttk.Label(position_window, text="Recomendación de Trading", font=("Helvetica", 16)).pack(pady=10)
    ttk.Label(position_window, text=f"Tipo de Posición: {suggestion['position_type']}", font=("Helvetica", 14)).pack(pady=5)
    
    entry_label = ttk.Label(position_window, text=f"Entrada: {suggestion['entry_price']}", font=("Helvetica", 14))
    entry_label.pack(pady=5)
    ttk.Button(position_window, text="Copiar Entrada", command=lambda: copy_to_clipboard(suggestion['entry_price'])).pack(pady=5)

    tp_label = ttk.Label(position_window, text=f"Take Profit: {suggestion['take_profit']}", font=("Helvetica", 14))
    tp_label.pack(pady=5)
    ttk.Button(position_window, text="Copiar Take Profit", command=lambda: copy_to_clipboard(suggestion['take_profit'])).pack(pady=5)

    sl_label = ttk.Label(position_window, text=f"Stop Loss: {suggestion['stop_loss']}", font=("Helvetica", 14))
    sl_label.pack(pady=5)
    ttk.Button(position_window, text="Copiar Stop Loss", command=lambda: copy_to_clipboard(suggestion['stop_loss'])).pack(pady=5)


# Función para manejar mensajes del WebSocket
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
                volume = float(kline['v'])
                new_data = pd.DataFrame([[time, open_price, high_price, low_price, close_price, volume]], columns=["time", "open", "high", "low", "close", "volume"])
                price_data = pd.concat([price_data, new_data]).drop_duplicates(subset=['time']).tail(500)
                update_plot()
                lbl_price.config(text=f"Último precio: {format_price(current_symbol, close_price)}")
    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")

# Suscribirse al símbolo y intervalo seleccionados
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

# Cancelar la suscripción al símbolo y intervalo actuales
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

# Manejar la selección de símbolo en la lista
def on_select(event):
    global current_symbol
    selection = lb.curselection()
    if selection:
        new_symbol = lb.get(selection)
        if current_symbol:
            unsubscribe_from_symbol(current_symbol, current_interval)
        current_symbol = new_symbol
        update_data_and_model()

# Manejar cambio de intervalo
def on_interval_change(event):
    global current_interval
    new_interval = interval_var.get()
    if current_symbol and new_interval != current_interval:
        unsubscribe_from_symbol(current_symbol, current_interval)
        current_interval = new_interval
        update_data_and_model()

# Mostrar ventana de progreso
def show_progress_window():
    global progress_window, progress_bar
    progress_window = tk.Toplevel(root)
    progress_window.title("Entrenando Modelo...")
    progress_window.geometry("300x100")
    ttk.Label(progress_window, text="Entrenando modelo, por favor espera...").pack(pady=10)
    progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
    progress_bar.pack(fill=tk.X, padx=20, pady=10)
    progress_bar.start()

# Cerrar ventana de progreso
def close_progress_window():
    if progress_window and progress_bar:
        progress_bar.stop()
        progress_window.destroy()

# Actualizar los datos y el modelo basado en el símbolo y el intervalo actuales
def update_data_and_model():
    global price_data, model, input_scaler, output_scaler
    show_progress_window()
    threading.Thread(target=update_data_and_model_thread).start()

def update_data_and_model_thread():
    global price_data, model, input_scaler, output_scaler
    limit = 1000  # Ajusta el límite aquí si necesitas más datos
    if 'd' in current_interval or 'w' in current_interval or 'M' in current_interval:
        limit = 2000  # Aumenta para intervalos más largos
    price_data = fetch_binance_historical_data(current_symbol, current_interval, limit=limit)
    price_data = add_technical_indicators(price_data)
    X, y, direction, input_scaler, output_scaler = prepare_data_with_indicators(price_data, window_size=60)
    model = train_model_with_indicators(X, y, direction)
    subscribe_to_symbol(current_symbol, current_interval)
    close_progress_window()

# Actualizar la lista de símbolos
def update_list(*args):
    search_term = search_var.get().upper()
    lb.delete(0, tk.END)
    for symbol in filter(lambda x: search_term in x, crypto_list):
        lb.insert(tk.END, symbol)

# Cargar la lista de criptomonedas
def load_crypto_list():
    global crypto_list
    crypto_list = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
        'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'SHIBUSDT'
    ]
    for symbol in crypto_list:
        lb.insert(tk.END, symbol)

# Ejecutar el WebSocket en un hilo
def run_websocket():
    global ws
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws",
        on_message=on_message,
        on_error=lambda ws, error: print(f"WebSocket error: {error}"),
        on_close=lambda ws, close_status_code, close_msg: print("WebSocket closed")
    )
    ws.run_forever()

# Actualizar el gráfico de precios
def update_plot():
    if not price_data.empty:
        ax.clear()
        mpf.plot(price_data.set_index('time'), type='candle', ax=ax, style='charles')
        ax.set_title(f"{current_symbol} Price")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.tight_layout()
        canvas.draw()

# Función para formatear el precio según los decimales del símbolo
def format_price(symbol, price):
    decimals = crypto_decimals.get(symbol, 2)
    return f"{price:.{decimals}f}"

# Función para sugerir una posición de trading
def on_suggest():
    suggestion = suggest_trading_position(model, input_scaler, output_scaler, price_data, current_symbol)
    show_trading_position(suggestion)

# Crear la ventana de configuración de indicadores
def open_settings_window():
    settings_window = tk.Toplevel(root)
    settings_window.title("Configuración de Indicadores Técnicos")

    tk.Checkbutton(settings_window, text="SMA", variable=indicators_selected['sma']).pack(anchor='w')
    tk.Checkbutton(settings_window, text="EMA", variable=indicators_selected['ema']).pack(anchor='w')
    tk.Checkbutton(settings_window, text="RSI", variable=indicators_selected['rsi']).pack(anchor='w')
    tk.Checkbutton(settings_window, text="Bollinger Bands", variable=indicators_selected['bollinger']).pack(anchor='w')
    tk.Checkbutton(settings_window, text="MACD", variable=indicators_selected['macd']).pack(anchor='w')
    tk.Checkbutton(settings_window, text="ATR", variable=indicators_selected['atr']).pack(anchor='w')

    tk.Button(settings_window, text="Aplicar", command=lambda: [update_data_and_model(), settings_window.destroy()]).pack(pady=10)

# Configuración de la ventana principal y widgets
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

btn_settings = ttk.Button(control_frame, text="Configuración de Indicadores", command=open_settings_window)
btn_settings.pack(pady=5)

load_crypto_list()

# Obtener la cantidad de decimales para cada símbolo
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
            decimals = len('{:.8f}'.format(tick_size).rstrip('0').split('.')[1])
            crypto_decimals[symbol] = decimals

get_symbol_decimals()

threading.Thread(target=run_websocket, daemon=True).start()

root.mainloop()
