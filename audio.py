"""
audio.py  –  Modulo de sintesis de voz (Bark TTS)
Convierte un archivo .txt generado por el OCR en un .mp3 reproducible.

Uso como modulo desde main.py:
    from audio import txt_a_mp3_async

Uso directo (linea de comandos):
    python audio.py ruta/al/texto.txt
"""

import gc
import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np

# ── Configuracion TTS ────────────────────────────────────────────────────────
SPEAKER       = "v2/es_speaker_6"
# TTS de frases largas: las divide en fragmentos de este tamaño (caracteres)
MAX_CHARS_PER_CHUNK = 180


# ── Estado del modelo Bark (carga perezosa) ──────────────────────────────────
_bark_loaded  = False
_bark_lock    = threading.Lock()


def _ensure_bark(progress_cb: Callable[[float, str], None] | None = None) -> bool:
    """Carga el modelo Bark la primera vez (thread-safe). Devuelve True si OK."""
    global _bark_loaded
    with _bark_lock:
        if _bark_loaded:
            return True
        try:
            import torch
            from bark import preload_models

            if progress_cb:
                progress_cb(0.05, "Cargando modelo Bark...")

            gc.collect()
            torch.cuda.empty_cache()

            # Guardar y parchar torch.load para compatibilidad de pesos
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            preload_models(
                text_use_small=True,
                coarse_use_small=True,
                fine_use_small=True,
                text_use_gpu=True,
            )
            torch.load = _orig
            _bark_loaded = True
            if progress_cb:
                progress_cb(0.15, "Modelo Bark listo.")
            return True
        except Exception as exc:
            print(f"[Audio] Error cargando Bark: {exc}")
            return False


def _leer_txt(txt_path: str) -> list[str]:
    """Lee el .txt y devuelve lista de fragmentos no vacíos."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lineas = [l.strip() for l in f if l.strip()]

    # Dividir lineas largas en fragmentos de MAX_CHARS_PER_CHUNK caracteres
    chunks: list[str] = []
    for linea in lineas:
        while len(linea) > MAX_CHARS_PER_CHUNK:
            # Cortar en el ultimo espacio dentro del limite
            corte = linea.rfind(" ", 0, MAX_CHARS_PER_CHUNK)
            if corte == -1:
                corte = MAX_CHARS_PER_CHUNK
            chunks.append(linea[:corte])
            linea = linea[corte:].strip()
        if linea:
            chunks.append(linea)
    return chunks


def _generar_audio(
    txt_path: str,
    mp3_path: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> bool:
    """
    Genera el .mp3 a partir del .txt.
    progress_cb(fraccion 0.0-1.0, mensaje):  llamado en el hilo de trabajo.
    Devuelve True si el archivo se creo correctamente.
    """
    try:
        import torch
        from bark import SAMPLE_RATE, generate_audio
        from scipy.io.wavfile import write as write_wav

        if not _ensure_bark(progress_cb):
            return False

        if not os.path.exists(txt_path):
            print(f"[Audio] No existe el archivo: {txt_path}")
            return False

        chunks = _leer_txt(txt_path)
        if not chunks:
            print("[Audio] El .txt esta vacio, no se genera audio.")
            return False

        total   = len(chunks)
        partes: list[np.ndarray] = []

        for i, texto in enumerate(chunks):
            frac_base = 0.15 + 0.80 * (i / total)
            if progress_cb:
                progress_cb(frac_base, f"Sintetizando {i+1}/{total}...")

            prompt = f"♪ {texto} ♪"
            print(f"[Audio] [{i+1}/{total}] {texto}")
            try:
                with torch.no_grad():
                    chunk = generate_audio(prompt, history_prompt=SPEAKER, text_temp=0.5)
                partes.append(chunk)
            except Exception as exc:
                print(f"[Audio] Error en fragmento {i+1}: {exc}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        if not partes:
            return False

        audio = np.concatenate(partes)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        audio_i16 = (audio * 32767).astype(np.int16)

        # Guardar WAV temporal y convertir a MP3 con pydub/ffmpeg
        wav_tmp = mp3_path.replace(".mp3", "_tmp.wav")
        write_wav(wav_tmp, SAMPLE_RATE, audio_i16)

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_wav(wav_tmp)
            seg.export(mp3_path, format="mp3", bitrate="128k")
            os.remove(wav_tmp)
            print(f"[Audio] MP3 guardado -> {mp3_path}")
        except Exception:
            # pydub/ffmpeg no disponible: dejar WAV renombrado
            import shutil
            wav_final = mp3_path.replace(".mp3", ".wav")
            shutil.move(wav_tmp, wav_final)
            print(f"[Audio] (pydub no disponible) WAV guardado -> {wav_final}")
            # Actualizar la ruta que devuelve para que main.py use la correcta
            mp3_path = wav_final

        if progress_cb:
            progress_cb(1.0, "Audio listo!")
        return True

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

    _txt  = sys.argv[1]
    _mp3  = sys.argv[2] if len(sys.argv) > 2 else _txt.replace(".txt", ".mp3")

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