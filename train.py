#!/usr/bin/env python3
"""
train.py  -  Entrenamiento con graficas de precision y error
             SIN TensorFlow  (PyTorch + scikit-learn + matplotlib)

Modelos entrenados
──────────────────
  1. OCR  (imagen → texto)  : CNN convolucional  (PyTorch)
     Reconoce caracteres alfanumericos (A-Z, 0-9) en imagenes 32x32.

  2. TTS  (texto → audio)   : MLP regresor       (PyTorch)
     Predice la duracion (segundos) del audio sintetizado a partir de
     rasgos del texto (longitud, silabas, puntuacion, etc.).

Graficas generadas en 'graficas/'
──────────────────────────────────
  ocr_perdida.png    Perdida entrenamiento/validacion  (OCR)
  ocr_precision.png  Precision entrenamiento/validacion (OCR)
  ocr_cer.png        Tasa de Error de Caracter          (OCR)
  tts_perdida.png    Perdida MSE entrenamiento/validacion (TTS)
  tts_mae.png        Error Absoluto Medio                 (TTS)
  tts_r2.png         Coeficiente de Determinacion R²      (TTS)
  resumen.png        Panel 2×3 con todas las graficas

Uso
───
  python train.py               # 50 epocas, datos sinteticos
  python train.py --demo        # alias del anterior
  python train.py --epochs 80   # numero de epocas personalizado
  python train.py --no-show     # guarda graficas sin abrir ventanas
"""

# ─── Parseo anticipado para configurar el backend de matplotlib ──────────────
import argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--no-show", action="store_true")
_pre_args, _ = _pre.parse_known_args()

import matplotlib
if _pre_args.no_show:
    matplotlib.use("Agg")

# ─── Resto de importaciones ──────────────────────────────────────────────────
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─── Configuracion global ─────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
GRAFICAS_DIR = BASE_DIR / "graficas"
GRAFICAS_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Incluimos mayusculas + minusculas + digitos para mayor dificultad (62 clases)
CHARS      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
NUM_CLASES = len(CHARS)   # 62 clases
IMG_SIZE   = 32
BATCH_OCR  = 32
BATCH_TTS  = 32
LR_OCR     = 5e-4
LR_TTS     = 5e-3

# Paleta de colores para las graficas
_C_TRAIN  = "#1976D2"   # azul
_C_VAL    = "#D32F2F"   # rojo
_C_ACCENT = "#388E3C"   # verde

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F5F5F5",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.fontsize":  10,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  OCR  ─  DATASET:  imagenes sinteticas de caracteres
# ═══════════════════════════════════════════════════════════════════════════════

def _cargar_fuente(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Carga la mejor fuente TrueType disponible en el sistema."""
    for path in [
        "arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\cour.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _generar_imagen(char: str) -> np.ndarray:
    """
    Renderiza 'char' en una imagen 32x32 en escala de grises con
    aumentacion agresiva para que el aprendizaje sea gradual:
      - Tamano de fuente aleatorio
      - Desplazamiento aleatorio (±4 px)
      - Rotacion aleatoria (±25 grados)
      - Escala aleatoria (80%-120%)
      - Ruido gaussiano fuerte
      - Desenfoque ocasional
      - Inversion de contraste ocasional
    Devuelve array float32 en [0, 1].
    """
    # Tamano de fuente variable para mayor variabilidad
    font_size = random.randint(IMG_SIZE - 10, IMG_SIZE - 2)
    canvas    = IMG_SIZE * 2   # lienzo grande para rotar sin cortar

    img  = Image.new("L", (canvas, canvas), color=255)
    draw = ImageDraw.Draw(img)
    font = _cargar_fuente(font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    cx   = (canvas - tw) // 2 + random.randint(-4, 4)
    cy   = (canvas - th) // 2 + random.randint(-4, 4)
    draw.text((cx, cy), char, fill=0, font=font)

    # Rotacion aleatoria
    angulo = random.uniform(-25, 25)
    img = img.rotate(angulo, fillcolor=255, resample=Image.BILINEAR)

    # Escala aleatoria -> recortar al centro
    escala = random.uniform(0.80, 1.20)
    nuevo  = int(canvas * escala)
    img    = img.resize((nuevo, nuevo), Image.BILINEAR)
    offset = (nuevo - IMG_SIZE) // 2
    offset = max(0, offset)
    img    = img.crop((offset, offset, offset + IMG_SIZE, offset + IMG_SIZE))
    img    = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # Desenfoque ocasional
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    arr = np.array(img, dtype=np.float32) / 255.0

    # Ruido gaussiano
    arr += np.random.normal(0.0, random.uniform(0.08, 0.22), arr.shape).astype(np.float32)

    # Inversion de contraste ocasional (texto blanco sobre negro)
    if random.random() < 0.20:
        arr = 1.0 - arr

    return np.clip(arr, 0.0, 1.0)


class CharDataset(Dataset):
    """Dataset sintetico de imagenes de caracteres alfanumericos."""

    def __init__(self, muestras_por_clase: int = 80):
        imgs, etiquetas = [], []
        for etiqueta, char in enumerate(CHARS):
            for _ in range(muestras_por_clase):
                imgs.append(_generar_imagen(char))
                etiquetas.append(etiqueta)

        indices = list(range(len(imgs)))
        random.shuffle(indices)
        self._imgs     = [imgs[i]      for i in indices]
        self._etiquetas = [etiquetas[i] for i in indices]

    def __len__(self):
        return len(self._etiquetas)

    def __getitem__(self, idx):
        img = torch.tensor(self._imgs[idx]).unsqueeze(0)   # (1, H, W)
        lbl = torch.tensor(self._etiquetas[idx], dtype=torch.long)
        return img, lbl


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  OCR  ─  MODELO:  CNN simple
# ═══════════════════════════════════════════════════════════════════════════════

class OCR_CNN(nn.Module):
    """
    Red convolucional ligera para clasificar caracteres 32×32 (62 clases).

    Arquitectura:
      Conv(1→16) + ReLU + MaxPool  → 16×16
      Conv(16→32) + ReLU + MaxPool → 8×8
      Conv(32→64) + ReLU + MaxPool → 4×4
      Flatten → Linear(1024→128) → Dropout(0.5) → Linear(128→NUM_CLASES)
    Sin BatchNorm para que la convergencia sea mas gradual.
    """

    def __init__(self, num_clases: int = NUM_CLASES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  OCR  ─  ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def entrenar_ocr(epochs: int) -> dict:
    """
    Entrena OCR_CNN y devuelve el historico de metricas por epoca:
      train_loss, val_loss, train_acc, val_acc, cer
    """
    print("\n" + "=" * 64)
    print("  [1/2]  OCR  —  Imagen a Texto  (CNN, PyTorch)")
    print("=" * 64)

    print("[OCR] Generando imagenes sinteticas...")
    dataset = CharDataset(muestras_por_clase=80)
    n_train = int(0.80 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[OCR] Entrenamiento={n_train}  Validacion={n_val}  "
          f"Clases={NUM_CLASES}  Dispositivo={DEVICE}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_OCR, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_OCR, shuffle=False, num_workers=0)

    modelo    = OCR_CNN().to(DEVICE)
    criterio  = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=LR_OCR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epochs)

    historico = {k: [] for k in ("train_loss", "val_loss",
                                  "train_acc",  "val_acc", "cer")}

    barra = tqdm(range(1, epochs + 1), desc="OCR Entrenamiento", unit="ep",
                 ncols=80, colour="blue")

    for epoch in barra:
        # ── Entrenamiento ──
        modelo.train()
        acc_loss = acc_cor = acc_tot = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizador.zero_grad()
            salida = modelo(imgs)
            perdida = criterio(salida, lbls)
            perdida.backward()
            optimizador.step()
            acc_loss += perdida.item() * imgs.size(0)
            acc_cor  += (salida.argmax(1) == lbls).sum().item()
            acc_tot  += imgs.size(0)

        train_loss = acc_loss / acc_tot
        train_acc  = acc_cor  / acc_tot

        # ── Validacion ──
        modelo.eval()
        acc_loss = acc_cor = acc_tot = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                salida   = modelo(imgs)
                perdida  = criterio(salida, lbls)
                acc_loss += perdida.item() * imgs.size(0)
                acc_cor  += (salida.argmax(1) == lbls).sum().item()
                acc_tot  += imgs.size(0)

        val_loss = acc_loss / acc_tot
        val_acc  = acc_cor  / acc_tot
        cer      = (1.0 - val_acc) * 100.0

        scheduler.step()

        historico["train_loss"].append(train_loss)
        historico["val_loss"].append(val_loss)
        historico["train_acc"].append(train_acc * 100.0)
        historico["val_acc"].append(val_acc  * 100.0)
        historico["cer"].append(cer)

        barra.set_postfix(
            loss=f"{val_loss:.3f}",
            acc=f"{val_acc*100:.1f}%",
            CER=f"{cer:.1f}%",
        )

    final_acc = historico["val_acc"][-1]
    final_cer = historico["cer"][-1]
    print(f"[OCR] Resultado final → Precision: {final_acc:.1f}%  |  CER: {final_cer:.1f}%")
    return historico


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TTS  ─  DATASET:  texto → duracion de habla
# ═══════════════════════════════════════════════════════════════════════════════

_VOCAB_ES = [
    "hola", "mundo", "mano", "robot", "gesto", "dedo", "indice",
    "dibujar", "lapiz", "borrar", "guardar", "salir", "limpiar",
    "activar", "desactivar", "reconocer", "procesar", "capturar",
    "texto", "audio", "camara", "sensor", "imagen", "fotografia",
    "universidad", "automatica", "digital", "laboratorio", "proyecto",
    "sistema", "control", "senal", "respuesta", "movimiento",
    "precisión", "velocidad", "exactitud", "calidad", "resultado",
]


def _silabas_es(palabra: str) -> int:
    """Conteo aproximado de silabas en espanol (grupos de vocales)."""
    vocales, cuenta, anterior = "aeiouáéíóúü", 0, False
    for ch in palabra.lower():
        if ch in vocales:
            if not anterior:
                cuenta += 1
            anterior = True
        else:
            anterior = False
    return max(1, cuenta)


def _rasgos_texto(texto: str) -> list[float]:
    """5 rasgos numericos de un texto."""
    palabras    = texto.split()
    n_chars     = len(texto)
    n_palabras  = len(palabras)
    n_silabas   = sum(_silabas_es(p) for p in palabras)
    n_puntuacion = sum(1 for c in texto if c in ".,;:!?¡¿")
    long_media  = n_chars / max(n_palabras, 1)
    return [float(n_chars), float(n_palabras), float(n_silabas),
            float(n_puntuacion), long_media]


def _duracion_real(rasgos: list[float]) -> float:
    """
    Duración simulada de audio (seg).
    Aprox. 2.5 silabas/s con variabilidad realista.
    """
    n_silabas   = rasgos[2]
    n_puntuacion = rasgos[3]
    base = n_silabas / 2.5 + n_puntuacion * 0.18
    return max(0.1, base + np.random.normal(0.0, 0.07))


def _construir_dataset_tts(n_muestras: int = 600):
    """Genera pares (rasgos_texto, duracion) de forma sintetica."""
    rng = np.random.default_rng(42)
    X, y = [], []
    for _ in range(n_muestras):
        n      = int(rng.integers(2, 14))
        palabras = random.choices(_VOCAB_ES, k=n)
        if rng.random() < 0.45:
            palabras[-1] += "."
        texto  = " ".join(palabras)
        rasgos = _rasgos_texto(texto)
        dur    = _duracion_real(rasgos)
        X.append(rasgos)
        y.append(dur)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TTS  ─  MODELO:  MLP regresor
# ═══════════════════════════════════════════════════════════════════════════════

class TTS_MLP(nn.Module):
    """MLP de 4 capas para regresion de duracion de habla."""

    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),  nn.ReLU(),
            nn.Linear(64,        128), nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(128,       64),  nn.ReLU(),
            nn.Linear(64,        1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x).squeeze(-1)


class RegresorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TTS  ─  ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def entrenar_tts(epochs: int) -> dict:
    """
    Entrena TTS_MLP y devuelve el historico de metricas por epoca:
      train_loss (MSE), val_loss (MSE), train_mae, val_mae, val_r2
    """
    print("\n" + "=" * 64)
    print("  [2/2]  TTS  —  Texto a Audio  (MLP, PyTorch)")
    print("=" * 64)

    X, y = _construir_dataset_tts(n_muestras=600)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    print(f"[TTS] Entrenamiento={len(X_tr)}  Validacion={len(X_val)}  "
          f"Rasgos=5  Dispositivo={DEVICE}")

    train_loader = DataLoader(RegresorDataset(X_tr,  y_tr),  batch_size=BATCH_TTS, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(RegresorDataset(X_val, y_val), batch_size=BATCH_TTS, shuffle=False, num_workers=0)

    modelo      = TTS_MLP().to(DEVICE)
    criterio    = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=LR_TTS, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.StepLR(
        optimizador, step_size=max(1, epochs // 4), gamma=0.6
    )

    historico = {k: [] for k in ("train_loss", "val_loss",
                                  "train_mae",  "val_mae", "val_r2")}

    barra = tqdm(range(1, epochs + 1), desc="TTS Entrenamiento", unit="ep",
                 ncols=80, colour="green")

    for epoch in barra:
        # ── Entrenamiento ──
        modelo.train()
        t_loss = t_mae = n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizador.zero_grad()
            pred  = modelo(xb)
            loss  = criterio(pred, yb)
            loss.backward()
            optimizador.step()
            t_loss += loss.item() * xb.size(0)
            t_mae  += (pred - yb).abs().sum().item()
            n      += xb.size(0)
        train_loss = t_loss / n
        train_mae  = t_mae  / n

        # ── Validacion ──
        modelo.eval()
        v_loss = v_mae = n = 0
        preds_all, true_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred  = modelo(xb)
                loss  = criterio(pred, yb)
                v_loss += loss.item() * xb.size(0)
                v_mae  += (pred - yb).abs().sum().item()
                preds_all.extend(pred.cpu().numpy().tolist())
                true_all.extend(yb.cpu().numpy().tolist())
                n      += xb.size(0)
        val_loss = v_loss / n
        val_mae  = v_mae  / n

        # R²
        p   = np.array(preds_all)
        t   = np.array(true_all)
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2  = 1.0 - ss_res / (ss_tot + 1e-10)

        scheduler.step()

        historico["train_loss"].append(train_loss)
        historico["val_loss"].append(val_loss)
        historico["train_mae"].append(train_mae)
        historico["val_mae"].append(val_mae)
        historico["val_r2"].append(r2)

        barra.set_postfix(
            MSE=f"{val_loss:.4f}",
            MAE=f"{val_mae:.4f}",
            R2=f"{r2:.3f}",
        )

    print(f"[TTS] Resultado final → MSE: {historico['val_loss'][-1]:.4f}  |  "
          f"MAE: {historico['val_mae'][-1]:.4f}  |  R²: {historico['val_r2'][-1]:.4f}")
    return historico


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  GRAFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def _guardar(fig: plt.Figure, nombre: str, show: bool) -> None:
    ruta = GRAFICAS_DIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"  [✓] {ruta.name}")
    if show:
        plt.show()
    plt.close(fig)


def _annotate_final(ax, valores: list[float], color: str) -> None:
    """Anotacion con el valor final de la curva."""
    ep_final = len(valores)
    ax.annotate(
        f"{valores[-1]:.2f}",
        xy=(ep_final, valores[-1]),
        xytext=(ep_final - len(valores) * 0.08, valores[-1]),
        fontsize=9,
        color=color,
        ha="right",
    )


def graficar_ocr(h: dict, epochs: int, show: bool) -> None:
    """Genera las 3 graficas del modelo OCR."""
    ep = list(range(1, epochs + 1))
    print("[Graficas OCR]")

    # 1. Perdida
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, h["train_loss"], color=_C_TRAIN, lw=2, label="Entrenamiento")
    ax.plot(ep, h["val_loss"],   color=_C_VAL,   lw=2, ls="--", label="Validacion")
    _annotate_final(ax, h["train_loss"], _C_TRAIN)
    _annotate_final(ax, h["val_loss"],   _C_VAL)
    ax.set_xlabel("Epoca");  ax.set_ylabel("Perdida (Cross-Entropy Loss)")
    ax.set_title("OCR — Imagen a Texto\nPerdida de entrenamiento y validacion")
    ax.legend();  fig.tight_layout()
    _guardar(fig, "ocr_perdida.png", show)

    # 2. Precision
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, h["train_acc"], color=_C_TRAIN, lw=2, label="Entrenamiento")
    ax.plot(ep, h["val_acc"],   color=_C_VAL,   lw=2, ls="--", label="Validacion")
    _annotate_final(ax, h["train_acc"], _C_TRAIN)
    _annotate_final(ax, h["val_acc"],   _C_VAL)
    ax.set_xlabel("Epoca");  ax.set_ylabel("Precision (%)")
    ax.set_title("OCR — Imagen a Texto\nPrecision de entrenamiento y validacion")
    ax.set_ylim(0, 105);  ax.legend();  fig.tight_layout()
    _guardar(fig, "ocr_precision.png", show)

    # 3. CER
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, h["cer"], color=_C_ACCENT, lw=2, label="CER (validacion)")
    ax.fill_between(ep, h["cer"], alpha=0.18, color=_C_ACCENT)
    _annotate_final(ax, h["cer"], _C_ACCENT)
    ax.set_xlabel("Epoca");  ax.set_ylabel("Tasa de Error de Caracter (%)")
    ax.set_title("OCR — Imagen a Texto\nTasa de Error de Caracter (CER)")
    ax.legend();  fig.tight_layout()
    _guardar(fig, "ocr_cer.png", show)


def graficar_tts(h: dict, epochs: int, show: bool) -> None:
    """Genera las 3 graficas del modelo TTS."""
    ep = list(range(1, epochs + 1))
    print("[Graficas TTS]")

    # 1. Perdida MSE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, h["train_loss"], color=_C_TRAIN, lw=2, label="Entrenamiento")
    ax.plot(ep, h["val_loss"],   color=_C_VAL,   lw=2, ls="--", label="Validacion")
    _annotate_final(ax, h["train_loss"], _C_TRAIN)
    _annotate_final(ax, h["val_loss"],   _C_VAL)
    ax.set_xlabel("Epoca");  ax.set_ylabel("Perdida (MSE)")
    ax.set_title("TTS — Texto a Audio\nPerdida de entrenamiento y validacion (MSE)")
    ax.legend();  fig.tight_layout()
    _guardar(fig, "tts_perdida.png", show)

    # 2. MAE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, h["train_mae"], color=_C_TRAIN, lw=2, label="Entrenamiento")
    ax.plot(ep, h["val_mae"],   color=_C_VAL,   lw=2, ls="--", label="Validacion")
    _annotate_final(ax, h["train_mae"], _C_TRAIN)
    _annotate_final(ax, h["val_mae"],   _C_VAL)
    ax.set_xlabel("Epoca");  ax.set_ylabel("Error Absoluto Medio (segundos)")
    ax.set_title("TTS — Texto a Audio\nError Absoluto Medio (MAE)")
    ax.legend();  fig.tight_layout()
    _guardar(fig, "tts_mae.png", show)

    # 3. R²
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, h["val_r2"], color=_C_ACCENT, lw=2, label="R² (validacion)")
    ax.axhline(1.0, color="gray", ls=":", lw=1.2, label="Optimo (R²=1)")
    ax.fill_between(ep, h["val_r2"], alpha=0.18, color=_C_ACCENT)
    _annotate_final(ax, h["val_r2"], _C_ACCENT)
    ax.set_xlabel("Epoca");  ax.set_ylabel("Coeficiente de Determinacion R²")
    ax.set_title("TTS — Texto a Audio\nCoeficiente de Determinacion R²")
    ax.set_ylim(-0.15, 1.15);  ax.legend();  fig.tight_layout()
    _guardar(fig, "tts_r2.png", show)


def graficar_resumen(ocr_h: dict, tts_h: dict, epochs: int, show: bool) -> None:
    """Panel 2×3 con el resumen completo de ambos modelos."""
    ep  = list(range(1, epochs + 1))
    fig, ejes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Resumen de Entrenamiento\nOCR (Imagen → Texto)  |  TTS (Texto → Audio)",
        fontsize=15, fontweight="bold",
    )

    # Fila 0 — OCR
    _plot = [
        (ejes[0, 0], "OCR — Perdida",      "Cross-Entropy",
         [(ocr_h["train_loss"], "Entrenamiento", _C_TRAIN, "-"),
          (ocr_h["val_loss"],   "Validacion",    _C_VAL,   "--")], None),
        (ejes[0, 1], "OCR — Precision",    "Acc (%)",
         [(ocr_h["train_acc"],  "Entrenamiento", _C_TRAIN, "-"),
          (ocr_h["val_acc"],    "Validacion",    _C_VAL,   "--")], (0, 105)),
        (ejes[0, 2], "OCR — CER",          "CER (%)",
         [(ocr_h["cer"],        "CER (val)",     _C_ACCENT, "-")], None),
        (ejes[1, 0], "TTS — Perdida (MSE)","MSE",
         [(tts_h["train_loss"], "Entrenamiento", _C_TRAIN, "-"),
          (tts_h["val_loss"],   "Validacion",    _C_VAL,   "--")], None),
        (ejes[1, 1], "TTS — MAE",          "MAE (seg)",
         [(tts_h["train_mae"],  "Entrenamiento", _C_TRAIN, "-"),
          (tts_h["val_mae"],    "Validacion",    _C_VAL,   "--")], None),
        (ejes[1, 2], "TTS — R²",           "R²",
         [(tts_h["val_r2"],     "R² (val)",      _C_ACCENT, "-")], (-0.15, 1.15)),
    ]

    for ax, titulo, ylabel, curvas, ylim in _plot:
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("Epoca", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        for valores, etiqueta, color, ls in curvas:
            ax.plot(ep, valores, color=color, lw=1.8, ls=ls, label=etiqueta)
            if ls == "-" and len(curvas) == 1:
                ax.fill_between(ep, valores, alpha=0.15, color=color)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9)

    print("[Graficas Resumen]")
    fig.tight_layout()
    _guardar(fig, "resumen.png", show)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento OCR + TTS con graficas (sin TensorFlow)"
    )
    parser.add_argument("--demo",    action="store_true",
                        help="Modo demo con datos sinteticos (activado por defecto)")
    parser.add_argument("--epochs",  type=int, default=50,
                        help="Numero de epocas (default: 50)")
    parser.add_argument("--no-show", action="store_true",
                        help="Solo guardar graficas, no abrir ventanas")
    args = parser.parse_args()

    show   = not args.no_show
    epochs = max(1, args.epochs)

    print("\n" + "=" * 64)
    print("  Entrenamiento OCR + TTS  —  SIN TensorFlow")
    print("=" * 64)
    print(f"  Dispositivo : {DEVICE}")
    print(f"  Epocas      : {epochs}")
    print(f"  Graficas    : {GRAFICAS_DIR}")
    print("=" * 64)

    t0 = time.perf_counter()

    ocr_historico = entrenar_ocr(epochs)
    tts_historico = entrenar_tts(epochs)

    print("\n" + "-" * 64)
    print("  Generando grafica resumen...")
    print("-" * 64)
    graficar_resumen(ocr_historico, tts_historico, epochs, show=show)

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 64}")
    print(f"  Listo en {elapsed:.1f}s  |  Graficas guardadas en: {GRAFICAS_DIR}")
    print(f"{'=' * 64}")
    for f in sorted(GRAFICAS_DIR.glob("*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
