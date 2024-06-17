# crypto_viewer.py
import tkinter as tk
from tkinter import ttk
import threading
import websocket
import json
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las variables de entorno
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Variable global para almacenar el símbolo de la criptomoneda seleccionada
current_symbol = None
ws = None  # WebSocket global para gestionar la conexión

# Función para manejar mensajes de WebSocket
def on_message(ws, message):
    try:
        data = json.loads(message)
        # Verifica si el mensaje es una lista y busca el símbolo actual
        if isinstance(data, list):
            for item in data:
                if item['s'] == current_symbol:
                    price = item['c']
                    lbl_price.config(text=f"{current_symbol}: {price} USD")
                    break
    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")

# Función para suscribirse a un símbolo específico
def subscribe_to_symbol(symbol):
    if ws and ws.sock and ws.sock.connected:
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [
                f"{symbol.lower()}@miniTicker"
            ],
            "id": 1
        }))
        print(f"Suscrito a {symbol}")

# Función para cancelar suscripción a un símbolo específico
def unsubscribe_from_symbol(symbol):
    if ws and ws.sock and ws.sock.connected:
        ws.send(json.dumps({
            "method": "UNSUBSCRIBE",
            "params": [
                f"{symbol.lower()}@miniTicker"
            ],
            "id": 1
        }))
        print(f"Cancelada la suscripción a {symbol}")

# Función que se llama cuando se selecciona una criptomoneda en la lista
def on_select(event):
    global current_symbol
    selection = lb.curselection()
    if selection:
        new_symbol = lb.get(selection)
        if current_symbol:
            unsubscribe_from_symbol(current_symbol)
        current_symbol = new_symbol
        subscribe_to_symbol(current_symbol)
        lbl_price.config(text=f"Esperando datos para {current_symbol}...")

# Función para actualizar la lista de criptomonedas según la búsqueda
def update_list(*args):
    search_term = search_var.get().upper()
    lb.delete(0, tk.END)
    for symbol in filter(lambda x: search_term in x, crypto_list):
        lb.insert(tk.END, symbol)

# Función para cargar la lista de criptomonedas
def load_crypto_list():
    global crypto_list
    crypto_list = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
        'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'SHIBUSDT'
    ]
    for symbol in crypto_list:
        lb.insert(tk.END, symbol)

# Crear y ejecutar el WebSocket en un hilo separado
def run_websocket():
    global ws
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/!miniTicker@arr",
        on_message=on_message,
        on_error=lambda ws, error: print(f"WebSocket error: {error}"),
        on_close=lambda ws, close_status_code, close_msg: print("WebSocket closed")
    )
    ws.run_forever()

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Crypto Price Viewer")

# Crear el marco izquierdo para la lista de criptomonedas y la barra de búsqueda
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Crear el marco para la búsqueda
search_frame = tk.Frame(left_frame)
search_frame.pack(fill=tk.X, padx=5, pady=5)

# Variable de búsqueda
search_var = tk.StringVar()
search_var.trace("w", update_list)

# Campo de búsqueda
search_entry = ttk.Entry(search_frame, textvariable=search_var)
search_entry.pack(fill=tk.X, padx=5, pady=5)

# Crear una lista de selección para las criptomonedas
lb = tk.Listbox(left_frame, height=15)
lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
lb.bind('<<ListboxSelect>>', on_select)

# Crear una barra de desplazamiento para la lista
scrollbar = tk.Scrollbar(left_frame, orient="vertical")
scrollbar.config(command=lb.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

lb.config(yscrollcommand=scrollbar.set)

# Crear el marco derecho para mostrar el precio
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Etiqueta para mostrar el precio de la criptomoneda seleccionada
lbl_price = ttk.Label(right_frame, text="Seleccione una criptomoneda", font=("Helvetica", 16))
lbl_price.pack(pady=20)

# Cargar la lista de criptomonedas
load_crypto_list()

# Iniciar el WebSocket en un hilo separado
threading.Thread(target=run_websocket, daemon=True).start()

root.mainloop()
