import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from lightkurve import search_lightcurvefile, LightCurve, search_lightcurve
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Button, Static, Select, Label
from rich.text import Text
from rich.markup import escape  # Para escapar el contenido y evitar markup

# ---------- CONFIGURACIÓN GENERAL ----------
TARGET = "TIC 307210830"
DATA_DIR = "./data"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "transit_cnn.pt")
META_PATH = os.path.join(MODEL_DIR, "transit_cnn_meta.json")
INPUT_LENGTH = 2000
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.001


# ---------- FUNCIONES GENERALES (sin UI) ----------
def log_console(message):
    """Función de log que imprime mensajes a consola."""
    from datetime import datetime
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    return full_message


def download_lightcurve(target):
    log_console(f"Descargando curva de luz para {target}...")
    filepath = os.path.join(DATA_DIR, f"{target.replace(' ', '_')}.csv")
    if os.path.exists(filepath):
        log_console(f"Curva ya descargada: {filepath}, omitiendo descarga.")
        return

    search_result = search_lightcurve(target, mission="TESS", author="SPOC")
    for r in search_result:
        obs_id = r.obsid
        cache_path = os.path.expanduser("~/.lightkurve/cache/mastDownload")
        if os.path.exists(cache_path):
            for root, _, files in os.walk(cache_path):
                for file in files:
                    if str(obs_id) in file:
                        full_path = os.path.join(root, file)
                        log_console(f"Eliminando archivo potencialmente corrupto: {full_path}")
                        os.remove(full_path)

    try:
        lc_collection = search_result.download_all()
        lc = lc_collection.stitch().normalize().remove_nans()
    except Exception as e:
        log_console(f"Error al descargar {target}: {e}")
        raise

    df = pd.DataFrame({"time": lc.time.value, "flux": lc.flux.value})
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(filepath, index=False)
    log_console(f"Curva de luz guardada en {filepath}")


def load_and_preprocess(filepath):
    log_console(f"Preprocesando archivo: {filepath}")
    df = pd.read_csv(filepath)
    lc = LightCurve(time=df["time"], flux=df["flux"])
    flattened = lc.flatten(window_length=401)
    flux = flattened.flux.value
    if len(flux) < INPUT_LENGTH:
        flux = np.pad(flux, (0, INPUT_LENGTH - len(flux)), mode="constant", constant_values=0)
    else:
        flux = flux[:INPUT_LENGTH]
    log_console(f"Longitud final del vector de flujo: {len(flux)}")
    return flux.astype(np.float32)


class ExoplanetDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        flux = load_and_preprocess(self.filepaths[idx])
        flux_tensor = torch.tensor(flux).unsqueeze(0)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return flux_tensor, label


class TransitCNN(nn.Module):
    def __init__(self, input_length):
        super(TransitCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        conv_out_size = (input_length - 4) // 2
        self.fc1 = nn.Linear(8 * conv_out_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def get_confirmed_tics():
    df = pd.read_csv(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+tic_id+from+pscomppars+where+discoverymethod='Transit'+and+disc_facility+like+'%25TESS%25'&format=csv"
    )
    df["tic_id"] = df["tic_id"].astype(str).str.extract(r"(\d+)")
    return df["tic_id"].dropna().astype(int).unique().tolist()


def get_false_positives():
    df = pd.read_csv(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+tid+from+toi+where+tfopwg_disp='FP'&format=csv"
    )
    return df["tid"].dropna().astype(int).unique().tolist()


def run_training(confirmed_tics, false_tics):
    files = []
    labels = []

    for tic in confirmed_tics:
        try:
            download_lightcurve(f"TIC {tic}")
            filepath = os.path.join(DATA_DIR, f"TIC_{tic}.csv")
            if os.path.exists(filepath):
                files.append(filepath)
                labels.append(1)
        except Exception as e:
            log_console(f"Falló la descarga para TIC {tic}: {e}")

    for tic in false_tics:
        try:
            download_lightcurve(f"TIC {tic}")
            filepath = os.path.join(DATA_DIR, f"TIC_{tic}.csv")
            if os.path.exists(filepath):
                files.append(filepath)
                labels.append(0)
        except Exception as e:
            log_console(f"Falló la descarga para TIC {tic}: {e}")

    log_console(f"Total de ejemplos cargados: {len(files)}")
    dataset = ExoplanetDataset(files, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransitCNN(INPUT_LENGTH)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        log_console(f"Epoch {epoch+1}/{EPOCHS} - Loss promedio: {running_loss/len(dataloader):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"input_length": INPUT_LENGTH}, f)
    log_console(f"Modelo guardado en: {MODEL_PATH}")
    log_console(f"Metadatos guardados en: {META_PATH}")


def predict(target):
    log_console(f"Realizando predicción para {target}...")
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    input_length = meta["input_length"]

    model = TransitCNN(input_length)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    filepath = os.path.join(DATA_DIR, f"{target}.csv")
    df = pd.read_csv(filepath)
    lc = LightCurve(time=df["time"], flux=df["flux"])
    try:
        flattened = lc.flatten(window_length=min(401, len(lc.time)))
    except Exception as e:
        log_console(f"Error flattening light curve: {e}")
        return None
    flux = flattened.flux.value
    if len(flux) < input_length:
        flux = np.pad(flux, (0, input_length - len(flux)), mode="constant", constant_values=0)
    else:
        flux = flux[:input_length]

    X = np.array([flux], dtype=np.float32)[:, np.newaxis, :]
    X_tensor = torch.tensor(X)
    with torch.no_grad():
        prediction = model(X_tensor)
        prob = prediction.item()

    log_console(f"Resultado: Probabilidad de tránsito para {target} = {prob:.2%}")
    log_console("Mostrando curva de luz suavizada con predicción...")
    flattened.plot(title=f"Curva de luz ({target}) - Probabilidad de tránsito: {prob:.2%}")
    plt.show()


# ---------- INTERFAZ CON TEXTUAL ----------
class TICSelectorApp(App):

    MAX_OPTIONS = 20

    def __init__(self):
        super().__init__()
        self.selected_confirmed = []
        self.selected_false = []
        self.log_text = ""  # Variable para acumular el log
        self.log_area = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Seleccioná TICs confirmados:")
        self.confirmed_select = Select(options=[])
        self.confirmed_display = Horizontal()
        yield self.confirmed_select
        yield self.confirmed_display

        yield Static("Seleccioná TICs falsos positivos:")
        self.false_select = Select(options=[])
        self.false_display = Horizontal()
        yield self.false_select
        yield self.false_display

        yield Button("Descargar y Entrenar", id="download_button")
        yield Button("Limpiar Logs", id="clear_logs")

        self.log_area = Static("Log de eventos:", id="log-area")
        self.log_area.styles.border = ("round", "yellow")
        yield self.log_area

        yield Footer()

    def update_log(self, message):
        """Actualiza el log interno y el widget log_area sin interpretar markup."""
        from datetime import datetime
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_message = f"{timestamp} {message}"
        self.log_text += full_message + "\n"
        # Escapeamos el contenido para evitar el parsing de markup
        self.log_area.update(Text(escape(self.log_text)))

    async def on_mount(self) -> None:
        self.update_log("On Mount")

        def generate_tic_select_options(tic_ids, max_options=20):
            return [(f"TIC {tic_id}", f"{tic_id}") for tic_id in tic_ids[:max_options]]

        confirmed_tics = await asyncio.to_thread(get_confirmed_tics)
        false_tics = await asyncio.to_thread(get_false_positives)

        self.update_log(f"Cargado {len(confirmed_tics)} TICs confirmados y {len(false_tics)} falsos positivos.")
        options_confirmed = generate_tic_select_options(confirmed_tics)
        options_false = generate_tic_select_options(false_tics)

        print("Opciones confirmadas:", options_confirmed)
        print("Opciones falsos positivos:", options_false)

        self.confirmed_select.options = options_confirmed
        self.false_select.options = options_false

        self.confirmed_select.refresh()
        self.false_select.refresh()

        self.update_log(f"Loaded {self.MAX_OPTIONS} options for both Select elements.")

    def on_select_changed(self, event: Select.Changed) -> None:
        self.update_log(f"Evento on_select_changed detectado. Valor recibido: {event.value}")
        tic = event.value
        if event.select is self.confirmed_select:
            if tic not in self.selected_confirmed:
                self.selected_confirmed.append(tic)
                self.confirmed_display.mount(Label(f"TIC {tic}"))
                self.update_log(f"TIC confirmado seleccionado: {tic}")
        elif event.select is self.false_select:
            if tic not in self.selected_false:
                self.selected_false.append(tic)
                self.false_display.mount(Label(f"TIC {tic}"))
                self.update_log(f"TIC falso positivo seleccionado: {tic}")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "download_button":
            self.update_log(f"Descargando y entrenando curvas de luz para {len(self.selected_confirmed)} TICs confirmados y {len(self.selected_false)} falsos positivos...")
            #run_training(self.selected_confirmed, self.selected_false)
            #predict(TARGET)
            #self.update_log(f"Entrenando con {len(self.selected_confirmed)} TICs confirmados y {len(self.selected_false)} falsos positivos...")
            #run_training(self.selected_confirmed, self.selected_false)
        elif event.button.id == "clear_logs":
            self.log_text = ""
            self.log_area.update(Text("Log de eventos:", escape=True))

# ---------- EJECUCIÓN ----------
if __name__ == "__main__":
    log_console("Inicio del selector interactivo de TICs para detección de exoplanetas")
    app = TICSelectorApp()
    app.run()
    log_console("Proceso finalizado.")
