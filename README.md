# Detección de Exoplanetas a partir de Curvas de Luz (TESS)

Este proyecto permite descargar, procesar y analizar curvas de luz astronómicas de la misión **TESS**, utilizando una **CNN (Red Neuronal Convolucional)** entrenada para detectar tránsitos planetarios. Incluye una interfaz interactiva en terminal usando `Textual` que permite seleccionar estrellas con planetas confirmados o falsos positivos, descargar los datos, entrenar el modelo y realizar predicciones.

## Características

* Descarga automática de curvas de luz desde el MAST (TESS/SPOC).
* Limpieza y normalización de datos con `lightkurve`.
* Red convolucional en PyTorch para clasificación binaria (tránsito / no tránsito).
* Interfaz TUI con `Textual` para selección de TICs y ejecución del entrenamiento.
* Visualización del resultado con `matplotlib`.

## Estructura del proyecto

```
├── data/               # Archivos CSV con curvas de luz descargadas
├── models/             # Modelos entrenados y metadatos
├── main.py             # Código principal (este archivo)
├── README.md
```

## Modelo

El modelo `TransitCNN` tiene la siguiente arquitectura:

* 1D Convolution (entrada: curva de luz)
* MaxPooling
* Fully Connected (Linear)
* Sigmoid (salida de probabilidad de tránsito)

Entrena sobre ejemplos confirmados y falsos positivos de exoplanetas observados por TESS.

## Requisitos

* Python 3.9+
* PyTorch
* lightkurve
* pandas, numpy, matplotlib
* textual, rich

Instalación:

```bash
pip install torch lightkurve pandas numpy matplotlib textual rich
```

## Cómo usar

### 1. Ejecutar la interfaz

```bash
python main.py
```

### 2. Seleccionar TICs

La interfaz te permitirá elegir hasta 20 TICs confirmados y 20 TICs marcados como falsos positivos desde el archivo público del **NASA Exoplanet Archive**.

### 3. Descargar datos y entrenar

Presiona el botón **"Descargar y Entrenar"**. Esto:

* Descargará las curvas de luz de cada TIC.
* Preprocesará y aplanará las curvas.
* Entrenará un modelo CNN con esos datos.

### 4. Predicción

Una vez entrenado el modelo, puedes ejecutar la función `predict("TIC 307210830")` desde el código para obtener la probabilidad de tránsito.

## Ejemplo de uso

```python
from main import predict
predict("TIC 307210830")
```

Esto generará una gráfica con la curva de luz suavizada y la probabilidad de que exista un tránsito planetario.

## Datos utilizados

* [NASA Exoplanet Archive - pscomppars](https://exoplanetarchive.ipac.caltech.edu/)
* [TESS Light Curves - MAST via lightkurve](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)

## Notas

* El sistema elimina automáticamente archivos corruptos en la caché de `lightkurve`.
* Los datos se normalizan y recortan/padean hasta una longitud estándar configurable (`INPUT_LENGTH`).
* Los logs se visualizan en tiempo real dentro de la interfaz de `Textual`.


