"""
audio.py  –  Modulo de sintesis de voz (Bark)
Convierte un archivo .txt generado por el OCR en un .mp3 reproducible
usando el modelo generativo Bark (requiere GPU, modelos locales).

Dependencias:
    pip install bark scipy pydub
    (pydub requiere ffmpeg en el PATH para exportar a MP3)

Uso como modulo desde main.py:
    from audio import txt_a_mp3_async

Uso directo (linea de comandos):
    python audio.py ruta/al/texto.txt
"""

import gc
import os
import tempfile
import threading
from typing import Callable

import numpy as np
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# ── Configuracion TTS ────────────────────────────────────────────────────────
# Voces disponibles en espanol: v2/es_speaker_0 .. v2/es_speaker_9
SPEAKER = "v2/es_speaker_6"

# Tamanio de modelo: True = small (rapido), False = large (mayor calidad)
USE_SMALL = True

_models_loaded = False
_models_lock   = threading.Lock()


def _cargar_modelos() -> None:
    """Carga los modelos de Bark una sola vez (thread-safe)."""
    global _models_loaded
    with _models_lock:
        if _models_loaded:
            return
        gc.collect()
        torch.cuda.empty_cache()

        # Parche necesario para versiones recientes de PyTorch
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

        print("[Audio] Cargando modelos Bark (GPU)...")
        preload_models(
            text_use_small=USE_SMALL,
            coarse_use_small=USE_SMALL,
            fine_use_small=USE_SMALL,
            text_use_gpu=True,
        )

        torch.load = _orig
        _models_loaded = True
        print("[Audio] Modelos listos.")


def _leer_txt(txt_path: str) -> str:
    """Lee el .txt y devuelve el contenido como una sola cadena."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lineas = [l.strip() for l in f if l.strip()]
    return " ".join(lineas)


def _wav_a_mp3(wav_path: str, mp3_path: str) -> None:
    """Convierte un .wav a .mp3 llamando a ffmpeg directamente via subprocess."""
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fallo (codigo {result.returncode}):\n"
            + result.stderr.decode(errors="replace")
        )


def _generar_audio(
    txt_path: str,
    mp3_path: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> bool:
    """Genera el .mp3 a partir del .txt usando Bark (sincrono)."""
    try:
        if not os.path.exists(txt_path):
            print(f"[Audio] No existe el archivo: {txt_path}")
            return False

        texto = _leer_txt(txt_path)
        if not texto:
            print("[Audio] El .txt esta vacio, no se genera audio.")
            return False

        print(f"[Audio] Texto: {texto[:120]}{'...' if len(texto) > 120 else ''}")

        if progress_cb:
            progress_cb(0.05, "Cargando modelos Bark...")

        _cargar_modelos()

        if progress_cb:
            progress_cb(0.3, "Generando audio con Bark...")

        print(f"[Audio] Sintetizando con Bark (speaker: {SPEAKER})...")
        with torch.no_grad():
            audio_array = generate_audio(texto, history_prompt=SPEAKER)

        gc.collect()
        torch.cuda.empty_cache()

        if progress_cb:
            progress_cb(0.8, "Convirtiendo a MP3...")

        # Normalizar y guardar como WAV temporal, luego convertir a MP3
        audio_norm = audio_array / (np.max(np.abs(audio_array)) + 1e-9)
        audio_int16 = (audio_norm * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_tmp = tmp.name

        try:
            write_wav(wav_tmp, SAMPLE_RATE, audio_int16)
            _wav_a_mp3(wav_tmp, mp3_path)
        finally:
            if os.path.exists(wav_tmp):
                os.remove(wav_tmp)

        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            print(f"[Audio] MP3 guardado -> {mp3_path}")
            if progress_cb:
                progress_cb(1.0, "Audio listo!")
            return True
        else:
            raise RuntimeError("El archivo MP3 generado esta vacio o no existe.")

    except Exception as exc:
        print(f"[Audio] Error general: {exc}")
        if progress_cb:
            progress_cb(-1.0, f"Error: {exc}")
        return False


def txt_a_mp3_async(
    txt_path: str,
    mp3_path: str,
    on_progress: Callable[[float, str], None] | None = None,
    on_done: Callable[[bool, str], None] | None = None,
) -> threading.Thread:
    """
    Lanza la generacion de audio en un hilo de fondo.

    Parametros:
        txt_path    – ruta al .txt generado por EasyOCR
        mp3_path    – ruta de salida del .mp3
        on_progress – callback(fraccion: float, msg: str)  0.0-1.0
                      fraccion=-1 indica error
        on_done     – callback(exito: bool, ruta_mp3: str)

    Devuelve el Thread iniciado.
    """
    def _worker():
        ok = _generar_audio(txt_path, mp3_path, on_progress)
        if on_done:
            on_done(ok, mp3_path)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def reproducir_mp3(mp3_path: str) -> None:
    """Reproduce el archivo de audio usando pygame o playsound (lo que este disponible)."""
    if not os.path.exists(mp3_path):
        print(f"[Audio] Archivo no encontrado: {mp3_path}")
        return
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        print(f"[Audio] Reproduciendo (pygame): {mp3_path}")
        return
    except Exception:
        pass
    try:
        from playsound import playsound
        threading.Thread(target=playsound, args=(mp3_path,), daemon=True).start()
        print(f"[Audio] Reproduciendo (playsound): {mp3_path}")
        return
    except Exception:
        pass
    # Fallback Windows: os.startfile
    try:
        os.startfile(mp3_path)
        print(f"[Audio] Abriendo con aplicacion predeterminada: {mp3_path}")
    except Exception as exc:
        print(f"[Audio] No se pudo reproducir: {exc}")


# ── Ejecucion directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python audio.py ruta/al/texto.txt [salida.mp3]")
        sys.exit(1)

    _txt = sys.argv[1]
    _mp3 = sys.argv[2] if len(sys.argv) > 2 else _txt.replace(".txt", ".mp3")

    def _cb(frac, msg):
        bar = int(frac * 30) if frac >= 0 else 0
        print(f"\r[{'#'*bar}{' '*(30-bar)}] {msg}          ", end="", flush=True)

    print(f"Convirtiendo: {_txt}  ->  {_mp3}")
    ok = _generar_audio(_txt, _mp3, _cb)
    print()
    if ok:
        print("Listo!")
    else:
        print("Fallo la generacion de audio.")
        sys.exit(1)