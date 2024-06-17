import tkinter as tk
from tkinter import ttk, messagebox
import threading
import websocket
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
import requests
import mplfinance as mpf
from matplotlib.dates import date2num

# Variables globales
current_symbol = None
current_interval = '1m'
ws = None
price_data = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
ai_modules = {}  # Diccionario para los modelos de IA por criptomoneda
bot_running = False  # Estado del bot

# Diccionario de precisiones decimales para cada criptomoneda
decimal_precisions = {
    'BTCUSDT': 2,
    'ETHUSDT': 2,
    'BNBUSDT': 2,
    'XRPUSDT': 5,
    'ADAUSDT': 5,
    'SOLUSDT': 3,
    'DOGEUSDT': 5,
    'DOTUSDT': 2,
    'MATICUSDT': 4,
    'SHIBUSDT': 8
}

# Función para obtener la precisión decimal de una criptomoneda
def get_decimal_precision(symbol):
    return decimal_precisions.get(symbol, 2)  # Por defecto, usa 2 decimales

# Inicializar el modelo de IA
class ScalpingAIModule:
    def __init__(self):
        self.price_data = pd.DataFrame(columns=["time", "close"])
        self.stop_loss = None
        self.take_profit = None
        self.position = None  # Puede ser "long" o None
        self.entry_price = None  # Precio de entrada
        self.entry_time = None  # Tiempo de entrada
        self.rsi_period = 14
        self.stop_loss_pct = 0.005  # 0.5% por defecto
        self.take_profit_pct = 0.01  # 1% por defecto
        self.trades = []  # Lista para guardar las operaciones
        self.last_trade_time = None  # Tiempo de la última operación
        self.trade_interval_seconds = 10  # Intervalo mínimo entre operaciones en segundos

    def update_price_data(self, time, close_price):
        new_data = pd.DataFrame([[time, close_price]], columns=["time", "close"])
        self.price_data = pd.concat([self.price_data, new_data]).drop_duplicates(subset=['time']).tail(self.rsi_period + 1)

    def calculate_rsi(self):
        if len(self.price_data) < self.rsi_period + 1:
            return None
        delta = self.price_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def decide_action(self):
        if not bot_running:
            return "Hold"  # No hace nada si el bot está detenido

        current_time = self.price_data['time'].iloc[-1]
        if self.last_trade_time is not None and (current_time - self.last_trade_time).total_seconds() < self.trade_interval_seconds:
            return "Hold"  # Esperar si no ha pasado el intervalo mínimo entre operaciones

        if len(self.price_data) < self.rsi_period + 1:
            return "Hold"  # No hay suficiente información

        current_price = self.price_data['close'].iloc[-1]
        rsi = self.calculate_rsi()

        if self.position == "long":
            if current_price <= self.stop_loss:
                profit = (current_price - self.entry_price)
                self.trades.append((current_symbol, 'Sell', current_time, current_price, 'Stop Loss', profit, (current_time - self.entry_time)))
                self.position = None
                self.last_trade_time = current_time
                return "Sell - Stop Loss"
            elif current_price >= self.take_profit:
                profit = (current_price - self.entry_price)
                self.trades.append((current_symbol, 'Sell', current_time, current_price, 'Take Profit', profit, (current_time - self.entry_time)))
                self.position = None
                self.last_trade_time = current_time
                return "Sell - Take Profit"
        else:
            if rsi < 30:
                # Compra si el RSI está en sobreventa
                self.position = "long"
                self.entry_price = current_price
                self.entry_time = current_time
                self.stop_loss = current_price * (1 - self.stop_loss_pct)
                self.take_profit = current_price * (1 + self.take_profit_pct)
                self.trades.append((current_symbol, 'Buy', current_time, current_price, 'Entry', 0, pd.Timedelta(seconds=0)))
                self.last_trade_time = current_time
                return "Buy"

        return "Hold"

def on_message(ws, message):
    global price_data, current_symbol
    try:
        data = json.loads(message)
        if 'e' in data and data['e'] == 'kline' and data['s'] == current_symbol:
            kline = data['k']
            if kline['i'] == current_interval:
                time = datetime.fromtimestamp(kline['t'] / 1000)
                close_price = float(kline['c'])
                precision = get_decimal_precision(current_symbol)
                formatted_price = f"{close_price:.{precision}f}"
                new_data = pd.DataFrame([[time, float(kline['o']), float(kline['h']), float(kline['l']), close_price]], 
                                         columns=["time", "open", "high", "low", "close"])
                price_data = pd.concat([price_data, new_data]).drop_duplicates(subset=['time']).tail(500)
                update_plot()
                # Actualizar y decidir con la IA para la criptomoneda actual
                if current_symbol not in ai_modules:
                    ai_modules[current_symbol] = ScalpingAIModule()
                ai_module = ai_modules[current_symbol]
                ai_module.update_price_data(time, close_price)
                action = ai_module.decide_action()
                ai_decision_label.config(text=f"AI Decision for {current_symbol}: {action}")
                current_price_label.config(text=f"Precio actual de {current_symbol}: {formatted_price} USD")
    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")

def subscribe_to_symbol(symbol, interval):
    global current_symbol
    current_symbol = symbol
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
        # Inicializar los datos del módulo de IA con los históricos
        if current_symbol not in ai_modules:
            ai_modules[current_symbol] = ScalpingAIModule()
        ai_module = ai_modules[current_symbol]
        ai_module.price_data = pd.DataFrame(columns=["time", "close"])  # Reiniciar el precio histórico
        for index, row in price_data.iterrows():
            ai_module.update_price_data(row['time'], row['close'])

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
        # Limpiar datos faltantes
        price_data_clean = price_data.dropna(subset=["open", "high", "low", "close"])
        ax.clear()
        mpf.plot(price_data_clean.set_index('time'), type='candle', ax=ax, style='charles')

        # Marcas para las decisiones de compra/venta
        if current_symbol in ai_modules:
            ai_module = ai_modules[current_symbol]
            for trade in ai_module.trades:
                symbol, action, time, price, reason, profit, duration = trade
                color = 'g' if 'Buy' in action else 'r'
                # Convertir tiempo a número flotante para anotaciones
                time_num = date2num(time)
                ax.annotate(action, xy=(time_num, price), xytext=(time_num, price),
                            arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='wedge'),
                            fontsize=10, color=color, horizontalalignment='left', verticalalignment='bottom')

        ax.set_title(f"{current_symbol} Price")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.tight_layout()
        canvas.draw()

def show_trades():
    # Crear ventana para mostrar el reporte
    report_window = tk.Toplevel(root)
    report_window.title("Reporte de Compras/Ventas")
    report_text = tk.Text(report_window, wrap='word', height=20, width=80)
    report_text.pack(padx=10, pady=10)

    # Añadir datos de las operaciones al reporte
    report_text.insert(tk.END, "Criptomoneda\tAcción\t\tPrecio\t\tTiempo\t\t\tRazón\t\tGanancia\t\tDuración\n")
    report_text.insert(tk.END, "------------\t------\t\t------\t\t------\t\t\t------\t\t--------\t\t--------\n")
    for symbol, ai_module in ai_modules.items():
        for trade in ai_module.trades:
            symbol, action, time, price, reason, profit, duration = trade
            profit_color = "green" if profit > 0 else "red"
            duration_str = str(duration).split('.')[0]  # Duración sin microsegundos
            report_text.insert(tk.END, f"{symbol}\t\t{action}\t\t{price:.{get_decimal_precision(symbol)}f}\t\t{time}\t{reason}\t\t", (profit_color,))
            report_text.insert(tk.END, f"{profit:.{get_decimal_precision(symbol)}f}\t\t{duration_str}\n")

    report_text.tag_configure("green", foreground="green")
    report_text.tag_configure("red", foreground="red")
    report_text.config(state=tk.DISABLED)

def toggle_bot():
    global bot_running
    bot_running = not bot_running
    bot_status_label.config(text=f"Bot Status: {'Running' if bot_running else 'Stopped'}")
    toggle_bot_button.config(text="Stop Bot" if bot_running else "Start Bot")

def configure_bot():
    if current_symbol not in ai_modules:
        messagebox.showerror("Error", "Seleccione una criptomoneda primero.")
        return
    
    # Crear ventana para configurar el bot
    config_window = tk.Toplevel(root)
    config_window.title("Configurar Bot")

    tk.Label(config_window, text="RSI Period:").pack(pady=5)
    rsi_entry = ttk.Entry(config_window)
    rsi_entry.insert(0, ai_modules[current_symbol].rsi_period)
    rsi_entry.pack(pady=5)

    tk.Label(config_window, text="Stop Loss %:").pack(pady=5)
    stop_loss_entry = ttk.Entry(config_window)
    stop_loss_entry.insert(0, ai_modules[current_symbol].stop_loss_pct)
    stop_loss_entry.pack(pady=5)

    tk.Label(config_window, text="Take Profit %:").pack(pady=5)
    take_profit_entry = ttk.Entry(config_window)
    take_profit_entry.insert(0, ai_modules[current_symbol].take_profit_pct)
    take_profit_entry.pack(pady=5)

    def save_config():
        try:
            ai_modules[current_symbol].rsi_period = int(rsi_entry.get())
            ai_modules[current_symbol].stop_loss_pct = float(stop_loss_entry.get())
            ai_modules[current_symbol].take_profit_pct = float(take_profit_entry.get())
            config_window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Por favor, introduzca valores válidos.")

    ttk.Button(config_window, text="Guardar", command=save_config).pack(pady=10)

root = tk.Tk()
root.title("Crypto Price Viewer")

# Maximiza la ventana al abrir
root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")

# Crear un PanedWindow para dividir la pantalla
paned_window = tk.PanedWindow(root, orient=tk.VERTICAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Frame superior (para el gráfico)
top_frame = tk.Frame(paned_window)
paned_window.add(top_frame)

# Frame inferior (para otros comandos y botones)
bottom_frame = tk.Frame(paned_window)
paned_window.add(bottom_frame)

# Configurar el gráfico en el frame superior
fig_width = 10
fig_height = 5

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
canvas = FigureCanvasTkAgg(fig, master=top_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Añadir widgets al frame inferior
control_frame = tk.Frame(bottom_frame)
control_frame.pack(fill=tk.BOTH, expand=True)

lbl_price = ttk.Label(control_frame, text="Seleccione una criptomoneda", font=("Helvetica", 16))
lbl_price.pack(pady=10)

current_price_label = ttk.Label(control_frame, text="Precio actual: --", font=("Helvetica", 14))
current_price_label.pack(pady=5)

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

# Botón para mostrar las compraventas
report_button = ttk.Button(control_frame, text="Ver Compraventas", command=show_trades)
report_button.pack(pady=5)

# Botón para iniciar/detener el bot
toggle_bot_button = ttk.Button(control_frame, text="Start Bot", command=toggle_bot)
toggle_bot_button.pack(pady=5)

# Label para mostrar el estado del bot
bot_status_label = ttk.Label(control_frame, text="Bot Status: Stopped", font=("Helvetica", 14))
bot_status_label.pack(pady=5)

# Botón para configurar el bot
configure_button = ttk.Button(control_frame, text="Configurar Bot", command=configure_bot)
configure_button.pack(pady=5)

# Label para mostrar la decisión de la IA
ai_decision_label = ttk.Label(control_frame, text="AI Decision: Hold", font=("Helvetica", 14))
ai_decision_label.pack(pady=5)

load_crypto_list()

threading.Thread(target=run_websocket, daemon=True).start()

root.mainloop()
