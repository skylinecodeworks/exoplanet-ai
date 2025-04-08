import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurvefile, LightCurve, search_lightcurve
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# ---------- PASO 1: DESCARGAR Y GUARDAR CURVA DE LUZ ----------
def download_lightcurve(target):
    print(f"📥 Descargando curva de luz para {target}...")
    search_result = search_lightcurve(target, mission="TESS", author="SPOC")

    # Verificación y limpieza manual de archivos corruptos
    for r in search_result:
        obs_id = r.obsid  # nombre único del archivo FITS
        cache_path = os.path.expanduser("~/.lightkurve/cache/mastDownload")
        if os.path.exists(cache_path):
            for root, _, files in os.walk(cache_path):
                for file in files:
                    if str(obs_id) in file:
                        full_path = os.path.join(root, file)
                        print(f"🧹 Eliminando archivo potencialmente corrupto: {full_path}")
                        os.remove(full_path)

    try:
        lc_collection = search_result.download_all()
        lc = lc_collection.stitch().normalize().remove_nans()
    except Exception as e:
        print(f"❌ Error al descargar {target}: {e}")
        raise

    df = pd.DataFrame({"time": lc.time.value, "flux": lc.flux.value})
    filepath = os.path.join(DATA_DIR, f"{target.replace(' ', '_')}.csv")
    print(f"✅ Curva de luz guardada en {filepath}")

# ---------- PASO 2: PREPROCESAMIENTO ----------
def load_and_preprocess(filepath):
    print(f"⚙️  Preprocesando archivo: {filepath}")
    df = pd.read_csv(filepath)
    lc = LightCurve(time=df["time"], flux=df["flux"])
    flattened = lc.flatten(window_length=401)
    flux = flattened.flux.value
    if len(flux) < INPUT_LENGTH:
        flux = np.pad(flux, (0, INPUT_LENGTH - len(flux)), mode="constant", constant_values=0)
    else:
        flux = flux[:INPUT_LENGTH]
    print(f"📐 Longitud final del vector de flujo: {len(flux)}")
    return flux.astype(np.float32)

# ---------- PASO 3: ENTRENAMIENTO DEL MODELO ----------
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

def train_model():
    print("🧠 Iniciando entrenamiento del modelo...")
    files = [os.path.join(DATA_DIR, f"{TARGET.replace(' ', '_')}.csv")]
    labels = [1]  # Simulación: positivo
    dataset = ExoplanetDataset(files, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransitCNN(INPUT_LENGTH)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"📉 Epoch {epoch+1}/{EPOCHS} - Loss promedio: {running_loss/len(dataloader):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"input_length": INPUT_LENGTH}, f)
    print(f"✅ Modelo guardado en: {MODEL_PATH}")
    print(f"🗃️  Metadatos guardados en: {META_PATH}")

# ---------- PASO 4: PREDICCIÓN Y VISUALIZACIÓN ----------
def predict(target):
    print(f"🔍 Realizando predicción para {target}...")
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    input_length = meta["input_length"]

    model = TransitCNN(input_length)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    filepath = os.path.join(DATA_DIR, f"{target}.csv")
    df = pd.read_csv(filepath)
    lc = LightCurve(time=df["time"], flux=df["flux"])
    flattened = lc.flatten(window_length=401)
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

    print(f"📊 Resultado: Probabilidad de tránsito para {target} = {prob:.2%}")
    print("📈 Mostrando curva de luz suavizada con predicción...")
    flattened.plot(title=f"Curva de luz ({target}) - Probabilidad de tránsito: {prob:.2%}")
    plt.show()

# ---------- EJECUCIÓN ----------
if __name__ == "__main__":
    print("🚀 Inicio del proceso de detección de exoplanetas")
    download_lightcurve(TARGET)
    train_model()
    predict(TARGET.replace(' ', '_'))
    print("🏁 Proceso finalizado.")
