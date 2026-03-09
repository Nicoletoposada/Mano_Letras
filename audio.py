"""
audio.py  –  Modulo de sintesis de voz (edge-tts)
Convierte un archivo .txt generado por el OCR en un .mp3 reproducible
usando Microsoft Edge TTS (no requiere GPU ni modelos locales).

Uso como modulo desde main.py:
    from audio import txt_a_mp3_async

Uso directo (linea de comandos):
    python audio.py ruta/al/texto.txt
"""

import asyncio
import os
import threading
from typing import Callable

# ── Configuracion TTS ────────────────────────────────────────────────────────
# Voces disponibles en espanol: es-ES-AlvaroNeural, es-ES-ElviraNeural,
# es-MX-DaliaNeural, es-MX-JorgeNeural, etc.
VOICE = "es-ES-AlvaroNeural"


def _leer_txt(txt_path: str) -> str:
    """Lee el .txt y devuelve el contenido como una sola cadena."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lineas = [l.strip() for l in f if l.strip()]
    return " ".join(lineas)


async def _generar_audio_async(
    txt_path: str,
    mp3_path: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> bool:
    """Genera el .mp3 a partir del .txt usando edge-tts (async)."""
    try:
        import edge_tts

        if not os.path.exists(txt_path):
            print(f"[Audio] No existe el archivo: {txt_path}")
            return False

        texto = _leer_txt(txt_path)
        if not texto:
            print("[Audio] El .txt esta vacio, no se genera audio.")
            return False

        print(f"[Audio] Sintetizando con edge-tts (voz: {VOICE})...")
        print(f"[Audio] Texto: {texto[:120]}{'...' if len(texto) > 120 else ''}")

        if progress_cb:
            progress_cb(0.1, "Conectando con servicio TTS...")

        communicate = edge_tts.Communicate(texto, VOICE)

        if progress_cb:
            progress_cb(0.3, "Generando audio...")

        await communicate.save(mp3_path)

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


def _generar_audio(
    txt_path: str,
    mp3_path: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> bool:
    """Wrapper sincrono sobre _generar_audio_async para usarlo desde hilos."""
    return asyncio.run(_generar_audio_async(txt_path, mp3_path, progress_cb))


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